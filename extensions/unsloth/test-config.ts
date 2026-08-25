/**
 * Unit tests for config.ts — run with: node test-config.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
	readConfig,
	writeConfig,
	upsertModel,
	removeModel,
	setDefaultModel,
	findConfiguredModelId,
	splitModelRef,
	buildLoadPayload,
	buildModelOverridePayload,
	buildSamplingParams,
	buildModelEntry,
	collectChatTemplates,
} from "./config.ts";

let passed = 0;
function test(name: string, fn: () => void) {
	try {
		fn();
		passed++;
		console.log(`ok   ${name}`);
	} catch (err) {
		console.error(`FAIL ${name}`);
		console.error(err);
		process.exitCode = 1;
	}
}

test("splitModelRef splits repo:quant, leaves bare ids alone", () => {
	assert.deepEqual(splitModelRef("unsloth/gemma-4-GGUF:UD-Q4_K_XL"), {
		modelPath: "unsloth/gemma-4-GGUF",
		variant: "UD-Q4_K_XL",
	});
	assert.deepEqual(splitModelRef("plain-model"), { modelPath: "plain-model" });
});

test("buildLoadPayload maps settings to Unsloth /load fields, omits unset", () => {
	const p = buildLoadPayload("org/model:Q8_0", {
		load: {
			maxSeqLength: 65536,
			cacheTypeKv: "q8_0",
			speculativeType: "off",
			specDraftNMax: 4,
			nParallel: 2,
			extraArgs: ["--flash-attn"],
		},
	});
	assert.deepEqual(p, {
		model_path: "org/model",
		gguf_variant: "Q8_0",
		max_seq_length: 65536,
		cache_type_kv: "q8_0",
		speculative_type: "off",
		spec_draft_n_max: 4,
		n_parallel: 2,
		llama_extra_args: ["--flash-attn"],
	});
});

test("DFlash plus n-gram cache produces one conflict-free server command", () => {
	const settings = {
		load: {
			speculation: {
				draft: {
					kind: "dflash",
					source: {
						kind: "huggingface",
						model: "incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M",
					},
					depth: 5,
				},
				ngram: { kind: "cache" },
			},
			extraArgs: [
				"--flash-attn",
				"--spec-type",
				"draft-mtp",
				"--model-draft",
				"old.gguf",
				"--spec-draft-n-max",
				"2",
				"--spec-type=draft-mtp",
				"--spec-draft-n-max=2",
			],
		},
	};
	assert.deepEqual(
		buildLoadPayload("ggml-org/Qwen3.8-27B-GGUF:Q4_K_M", settings),
		{
			model_path: "ggml-org/Qwen3.8-27B-GGUF",
			gguf_variant: "Q4_K_M",
			speculative_type: "dflash",
			spec_draft_n_max: 5,
			llama_extra_args: [
				"--flash-attn",
				"--spec-draft-hf",
				"incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M",
				"--spec-type",
				"draft-dflash,ngram-cache",
				"--spec-draft-n-max",
				"5",
			],
		},
	);
});

test("DFlash can use a file already on the Unsloth computer", () => {
	const payload = buildLoadPayload("org/model:Q4", {
		load: {
			speculation: {
				draft: {
					kind: "dflash",
					source: { kind: "file", path: "C:\\models\\draft.gguf" },
					depth: 7,
				},
				ngram: { kind: "none" },
			},
		},
	});
	assert.deepEqual(payload.llama_extra_args, [
		"--spec-draft-model",
		"C:\\models\\draft.gguf",
		"--spec-type",
		"draft-dflash",
		"--spec-draft-n-max",
		"7",
	]);
});

test("n-gram cache alone does not also tell Unsloth to turn speculation off", () => {
	const payload = buildLoadPayload("org/model", {
		load: {
			speculation: {
				draft: { kind: "none" },
				ngram: { kind: "cache" },
			},
		},
	});
	assert.equal(payload.speculative_type, "ngram");
	assert.deepEqual(payload.llama_extra_args, ["--spec-type", "ngram-cache"]);
});

test("MTP plus n-gram mod uses Unsloth's visible native controls", () => {
	const payload = buildLoadPayload("org/model", {
		load: {
			speculation: {
				draft: { kind: "mtp", depth: 2 },
				ngram: { kind: "mod", match: 20, min: 32, max: 48 },
			},
		},
	});
	assert.equal(payload.speculative_type, "mtp+ngram");
	assert.equal(payload.spec_draft_n_max, 2);
	assert.deepEqual(payload.llama_extra_args, [
		"--spec-ngram-mod-n-match",
		"20",
		"--spec-ngram-mod-n-min",
		"32",
		"--spec-ngram-mod-n-max",
		"48",
	]);
});

test("buildModelOverridePayload maps per-model server settings", () => {
	assert.deepEqual(
		buildModelOverridePayload("org/model:Q8_0", {
			chatTemplate: { id: "custom", name: "Custom", content: "CHAT" },
			load: {
				maxSeqLength: 65536,
				cacheTypeKv: "q8_0",
				speculativeType: "mtp",
				specDraftNMax: 4,
				nParallel: 2,
				nBatch: 4096,
				nUBatch: 1024,
				extraArgs: ["--flash-attn"],
			},
		}),
		{
			model_id: "org/model:Q8_0",
			remove: false,
			chat_template_override: "CHAT",
			max_seq_length: 65536,
			kv_cache_dtype: "q8_0",
			speculative_type: "mtp",
			spec_draft_n_max: 4,
			n_parallel: 2,
			n_batch: 4096,
			n_ubatch: 1024,
			llama_extra_args: ["--flash-attn"],
		},
	);
});

test("buildLoadPayload sends the selected chat template override", () => {
	assert.deepEqual(
		buildLoadPayload("org/model:Q4", {
			chatTemplate: {
				id: "hf:org/source:default",
				name: "Source default",
				content: "{% for message in messages %}CHAT{% endfor %}",
			},
		}),
		{
			model_path: "org/model",
			gguf_variant: "Q4",
			chat_template_override: "{% for message in messages %}CHAT{% endfor %}",
		},
	);
});

test("buildLoadPayload with model default sends no template override", () => {
	assert.deepEqual(buildLoadPayload("org/model", {}), {
		model_path: "org/model",
	});
});

test("buildModelOverridePayload with model defaults keeps the server entry without stale flags", () => {
	assert.deepEqual(buildModelOverridePayload("org/model", {}), {
		model_id: "org/model",
		remove: false,
		llama_extra_args: [],
	});
});

test("buildSamplingParams uses OpenAI wire names", () => {
	assert.deepEqual(
		buildSamplingParams({
			temperature: 0.6,
			topP: 0.95,
			topK: 20,
			minP: 0,
			repeatPenalty: 1.05,
			seed: 42,
		}),
		{
			temperature: 0.6,
			top_p: 0.95,
			top_k: 20,
			min_p: 0,
			repetition_penalty: 1.05,
			seed: 42,
		},
	);
	assert.equal(buildSamplingParams({}), undefined);
	assert.equal(buildSamplingParams(undefined), undefined);
});

test("buildModelEntry includes thinking compat, level map, samplingParams", () => {
	const e = buildModelEntry("org/m:Q4", {
		contextWindow: 64000,
		thinking: {
			reasoning: true,
			compat: {
				thinkingFormat: "qwen-chat-template",
				supportsReasoningEffort: false,
			},
		},
		sampling: { temperature: 0.7, topP: 0.8 },
	}) as any;
	assert.equal(e.id, "org/m:Q4");
	assert.equal(e.reasoning, true);
	assert.equal(e.contextWindow, 64000);
	assert.equal(e.compat.thinkingFormat, "qwen-chat-template");
	assert.deepEqual(e.samplingParams, { temperature: 0.7, top_p: 0.8 });
});

test("collectChatTemplates builds a reusable library across models without duplicates", () => {
	const shared = { id: "shared", name: "Shared", content: "SHARED" };
	const other = { id: "other", name: "Other", content: "OTHER" };
	const templates = collectChatTemplates({
		providers: {
			unsloth: {
				baseUrl: "http://x/v1",
				apiKey: "k",
				models: {
					a: { chatTemplate: shared, chatTemplates: [shared] },
					b: { chatTemplate: other, chatTemplates: [shared, other] },
				},
			},
		},
	});
	assert.deepEqual(templates, [shared, other]);
});

test("findConfiguredModelId matches Unsloth's unqualified active model to its saved quant", () => {
	const provider = {
		baseUrl: "http://x/v1",
		apiKey: "k",
		models: { "org/model:Q4_K_M": {}, "org/other:Q8_0": {} },
	};
	assert.equal(findConfiguredModelId(provider, "org/model"), "org/model:Q4_K_M");
	assert.equal(
		findConfiguredModelId(provider, "org/other:Q8_0"),
		"org/other:Q8_0",
	);
	assert.equal(findConfiguredModelId(provider, "missing"), undefined);
});

test("config store: round-trip, upsert, remove", () => {
	const dir = mkdtempSync(join(tmpdir(), "unsloth-cfg-"));
	const path = join(dir, "unsloth.json");
	let cfg = readConfig(path);
	assert.deepEqual(cfg, { providers: {} });

	upsertModel(
		cfg,
		"unsloth",
		{ baseUrl: "http://x/v1", apiKey: "k" },
		"org/m:Q4",
		{ sampling: { temperature: 0.5 } },
	);
	assert.equal(setDefaultModel(cfg, "unsloth", "org/m:Q4"), true);
	writeConfig(cfg, path);

	cfg = readConfig(path);
	assert.equal(cfg.providers.unsloth.baseUrl, "http://x/v1");
	assert.equal(
		cfg.providers.unsloth.models["org/m:Q4"].sampling?.temperature,
		0.5,
	);
	assert.equal(cfg.providers.unsloth.defaultModelId, "org/m:Q4");

	assert.equal(removeModel(cfg, "unsloth", "org/m:Q4"), true);
	assert.equal(cfg.providers.unsloth.defaultModelId, undefined);
	assert.equal(removeModel(cfg, "unsloth", "org/m:Q4"), false);
	assert.deepEqual(Object.keys(cfg.providers.unsloth.models), []);
	rmSync(dir, { recursive: true, force: true });
});

console.log(
	`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`,
);
