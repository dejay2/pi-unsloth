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
	splitModelRef,
	buildLoadPayload,
	buildSamplingParams,
	buildModelEntry,
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

test("buildLoadPayload with no settings sends just model_path", () => {
	assert.deepEqual(buildLoadPayload("org/model", {}), { model_path: "org/model" });
});

test("buildSamplingParams uses OpenAI wire names", () => {
	assert.deepEqual(buildSamplingParams({ temperature: 0.6, topP: 0.95, topK: 20, minP: 0, repeatPenalty: 1.05, seed: 42 }), {
		temperature: 0.6,
		top_p: 0.95,
		top_k: 20,
		min_p: 0,
		repetition_penalty: 1.05,
		seed: 42,
	});
	assert.equal(buildSamplingParams({}), undefined);
	assert.equal(buildSamplingParams(undefined), undefined);
});

test("buildModelEntry includes thinking compat, level map, samplingParams", () => {
	const e = buildModelEntry("org/m:Q4", {
		contextWindow: 64000,
		thinking: {
			reasoning: true,
			compat: { thinkingFormat: "qwen-chat-template", supportsReasoningEffort: false },
		},
		sampling: { temperature: 0.7, topP: 0.8 },
	}) as any;
	assert.equal(e.id, "org/m:Q4");
	assert.equal(e.reasoning, true);
	assert.equal(e.contextWindow, 64000);
	assert.equal(e.compat.thinkingFormat, "qwen-chat-template");
	assert.deepEqual(e.samplingParams, { temperature: 0.7, top_p: 0.8 });
});

test("config store: round-trip, upsert, remove", () => {
	const dir = mkdtempSync(join(tmpdir(), "unsloth-cfg-"));
	const path = join(dir, "unsloth.json");
	let cfg = readConfig(path);
	assert.deepEqual(cfg, { providers: {} });

	upsertModel(cfg, "unsloth", { baseUrl: "http://x/v1", apiKey: "k" }, "org/m:Q4", { sampling: { temperature: 0.5 } });
	writeConfig(cfg, path);

	cfg = readConfig(path);
	assert.equal(cfg.providers.unsloth.baseUrl, "http://x/v1");
	assert.equal(cfg.providers.unsloth.models["org/m:Q4"].sampling?.temperature, 0.5);

	assert.equal(removeModel(cfg, "unsloth", "org/m:Q4"), true);
	assert.equal(removeModel(cfg, "unsloth", "org/m:Q4"), false);
	assert.deepEqual(Object.keys(cfg.providers.unsloth.models), []);
	rmSync(dir, { recursive: true, force: true });
});

console.log(`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`);
