/**
 * Unit tests for wizard.ts — run with: node test-wizard.ts
 */
import assert from "node:assert/strict";
import {
	parseNumber,
	parseInt_,
	parseIntInRange,
	normalizeBaseUrl,
	providerSetup,
	pickModel,
	settingsWizard,
	chatTemplateWizard,
	type WizardInteraction,
} from "./wizard.ts";
import type { ReasoningInfo } from "./thinking.ts";

let passed = 0;
async function test(name: string, fn: () => void | Promise<void>) {
	try {
		await fn();
		passed++;
		console.log(`ok   ${name}`);
	} catch (err) {
		console.error(`FAIL ${name}`);
		console.error(err);
		process.exitCode = 1;
	}
}

function scripted(answers: { texts?: string[]; selects?: (string | undefined)[] }) {
	const texts = [...(answers.texts ?? [])];
	const selects = [...(answers.selects ?? [])];
	const progress: string[] = [];
	const selectCalls: { message: string; options: { id: string; label: string }[] }[] = [];
	const ui: WizardInteraction = {
		async text(message) {
			const next = texts.shift();
			if (next === undefined) throw new Error(`unexpected text prompt: ${message}`);
			return next;
		},
		async select(message, options) {
			selectCalls.push({ message, options });
			if (selects.length === 0) throw new Error(`unexpected select: ${message}`);
			return selects.shift();
		},
		progress(m) {
			progress.push(m);
		},
	};
	return { ui, progress, selectCalls };
}

await test("normalizeBaseUrl adds scheme and /v1, strips trailing slashes", () => {
	assert.equal(normalizeBaseUrl("localhost:8888"), "http://localhost:8888/v1");
	assert.equal(normalizeBaseUrl("http://h:8888/"), "http://h:8888/v1");
	assert.equal(normalizeBaseUrl("https://h/v1"), "https://h/v1");
	assert.equal(normalizeBaseUrl("https://h/v1/"), "https://h/v1");
});

await test("parseNumber/parseInt_: empty = undefined, invalid throws", () => {
	assert.equal(parseNumber(""), undefined);
	assert.equal(parseNumber("  "), undefined);
	assert.equal(parseNumber("0.6"), 0.6);
	assert.throws(() => parseNumber("abc"));
	assert.equal(parseInt_("3"), 3);
	assert.throws(() => parseInt_("3.5"));
});

await test("parseIntInRange rejects draft depths outside the supported range", () => {
	assert.equal(parseIntInRange("5", 1, 16, "Draft depth"), 5);
	assert.equal(parseIntInRange("", 1, 16, "Draft depth"), undefined);
	assert.throws(() => parseIntInRange("0", 1, 16, "Draft depth"), /between 1 and 16/);
	assert.throws(() => parseIntInRange("17", 1, 16, "Draft depth"), /between 1 and 16/);
});

await test("providerSetup returns normalized URL + key", async () => {
	const { ui } = scripted({ texts: ["100.106.5.124:8888", "sk-unsloth-abc"] });
	const out = await providerSetup(ui);
	assert.deepEqual(out, { baseUrl: "http://100.106.5.124:8888/v1", apiKey: "sk-unsloth-abc" });
});

await test("providerSetup rejects empty key", async () => {
	const { ui } = scripted({ texts: ["h:8888", "  "] });
	await assert.rejects(() => providerSetup(ui), /API key is required/);
});

await test("pickModel returns the chosen model", async () => {
	const { ui } = scripted({ selects: ["a:Q4"] });
	const m = await pickModel(ui, [{ id: "a:Q4" }, { id: "a:Q8" }]);
	assert.equal(m.id, "a:Q4");
});

await test("chat template: model default clears the current replacement but keeps history", async () => {
	const old = { id: "old", name: "Old template", content: "OLD" };
	const { ui } = scripted({ selects: ["__default"] });
	const result = await chatTemplateWizard(ui, old, [old], [old]);
	assert.equal(result.selected, undefined);
	assert.deepEqual(result.history, [old]);
});

await test("chat template: current model templates are offered directly", async () => {
	const one = { id: "one", name: "Template one", content: "ONE" };
	const two = { id: "two", name: "Template two", content: "TWO" };
	const { ui } = scripted({ selects: ["two"] });
	const result = await chatTemplateWizard(ui, one, [one, two], [one, two]);
	assert.deepEqual(result.selected, two);
});

await test("chat template: all-model library can be browsed and adds the choice to model history", async () => {
	const shared = { id: "shared", name: "Shared template", content: "SHARED" };
	const { ui } = scripted({ selects: ["__browse", "shared"] });
	const result = await chatTemplateWizard(ui, undefined, [], [shared]);
	assert.deepEqual(result.selected, shared);
	assert.deepEqual(result.history, [shared]);
});

await test("chat template: Hugging Face preselects default and allows an alternative", async () => {
	const normal = { id: "hf:org/m:default", name: "org/m — default", content: "NORMAL", isDefault: true };
	const tools = { id: "hf:org/m:tool_use", name: "org/m — tool_use", content: "TOOLS", isDefault: false };
	const { ui } = scripted({ texts: ["https://huggingface.co/org/m"], selects: ["__huggingface", tools.id] });
	const result = await chatTemplateWizard(ui, undefined, [], [], async () => [normal, tools]);
	assert.deepEqual(result.selected, tools);
	assert.deepEqual(result.history, [normal, tools]);
});

await test("chat template: pasted text is named, saved, and selected", async () => {
	const { ui } = scripted({ texts: ["My template", "{% for message in messages %}X{% endfor %}"], selects: ["__paste"] });
	const result = await chatTemplateWizard(ui, undefined, [], []);
	assert.equal(result.selected?.name, "My template");
	assert.equal(result.selected?.content, "{% for message in messages %}X{% endfor %}");
	assert.deepEqual(result.history, [result.selected]);
});

const detected: ReasoningInfo = {
	style: "enable_thinking_effort",
	levels: ["high", "max"],
	supportsPreserveThinking: false,
	activeModel: "org/m",
};

await test("settingsWizard: full answers flow into all settings groups", async () => {
	const { ui, progress, selectCalls } = scripted({
		texts: [
			"65536", // context
			"4",     // spec draft n
			"2",     // parallel
			"--flash-attn", // extra args
			"0.7", "0.8", "20", "0.05", "0.1", "0.0", "1.1", "", // sampling (temp, top_p, top_k, min_p, presence, frequency, rep, seed)
			"0.6", "0.95", "20", "0.0", "0.0", "0.0", "1.0", "", // thinking sampling (custom)
		],
		selects: ["q8_0", "mtp", "none", "yes", "custom"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected);
	assert.equal(selectCalls[3].message, "Enable shared context for parallel slots?");
	assert.deepEqual(selectCalls[3].options.map((option) => option.id), ["yes", "no"]);
	assert.equal(s.contextWindow, 65536);
	assert.equal(s.load?.cacheTypeKv, "q8_0");
	assert.deepEqual(s.load?.speculation, {
		draft: { kind: "mtp", depth: 4 },
		ngram: { kind: "none" },
	});
	assert.equal(s.load?.nParallel, 2);
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn", "--kv-unified"]);
	assert.equal(s.sampling?.temperature, 0.7);
	assert.equal(s.sampling?.minP, 0.05);
	assert.equal(s.sampling?.presencePenalty, 0.1);
	assert.equal(s.samplingThinking?.temperature, 0.6);
	assert.equal(s.samplingThinking?.topP, 0.95);
	// thinking auto-configured from detection (loaded model)
	assert.equal(s.thinking?.reasoning, true);
	assert.equal(s.thinking?.thinkingLevelMap?.max, "max");
	assert.ok(progress.some((p) => p.includes("enable_thinking_effort")));
});

await test("settingsWizard: DFlash warns that Qwen3.8 DFlash2 needs the special llama.cpp build", async () => {
	const { ui, progress } = scripted({
		texts: [
			"", // context
			"5", // DFlash depth
			"incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M", // helper
			"1", // parallel
			"", // extra args
			"", "", "", "", "", "", "", "", // normal sampling
		],
		selects: ["server default", "dflash", "huggingface", "cache", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected);
	assert.deepEqual(s.load?.speculation, {
		draft: {
			kind: "dflash",
			source: { kind: "huggingface", model: "incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M" },
			depth: 5,
		},
		ngram: { kind: "cache" },
	});
	const warning = progress.find((message) => message.includes("PR #27342"));
	assert.ok(warning);
	assert.match(warning, /normal llama\.cpp/i);
	assert.match(warning, /github\.com\/ggml-org\/llama\.cpp\/pull\/27342/);
});

await test("settingsWizard: parallel one skips the shared-context question", async () => {
	const { ui } = scripted({
		texts: ["", "", "1", "", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "server-default", "none", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected, {
		load: { extraArgs: ["-kvu", "--kv-unified", "--flash-attn"] },
	});
	assert.equal(s.load?.nParallel, 1);
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn"]);
});

await test("settingsWizard: choosing no removes a previously saved shared-context flag", async () => {
	const { ui } = scripted({
		texts: ["", "", "2", "", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "server-default", "none", "no", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected, {
		load: { nParallel: 2, extraArgs: ["--kv-unified", "-kvu", "--flash-attn"] },
	});
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn"]);
});

await test("settingsWizard: choosing yes replaces aliases and duplicates with one canonical flag", async () => {
	const { ui } = scripted({
		texts: ["", "", "2", "", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "server-default", "none", "yes", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected, {
		load: { nParallel: 2, extraArgs: ["-kvu", "--kv-unified", "--flash-attn"] },
	});
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn", "--kv-unified"]);
});

await test("settingsWizard: empty answers keep defaults, 'same' skips thinking sampling", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "server-default", "none", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected);
	assert.equal(s.contextWindow, undefined); // model carried none
	assert.deepEqual(s.load, {
		speculation: { draft: { kind: "server-default" }, ngram: { kind: "none" } },
	});
	assert.equal(s.sampling?.temperature, 0.7); // default
	assert.equal(s.samplingThinking, undefined);
});

await test("settingsWizard: non-loaded model without detection gets reasoning=false, no thinking question", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "server-default", "none"],
	});
	const s = await settingsWizard(ui, { id: "org/other:Q4" }, detected);
	assert.deepEqual(s.thinking, { reasoning: false });
});

await test("settingsWizard: 'recommended' thinking sampling preset", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "server-default", "none", "recommended"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected);
	assert.deepEqual(s.samplingThinking, { temperature: 0.6, topP: 0.95, topK: 20, minP: 0 });
});

const existingSettings = {
	contextWindow: 65536,
	thinking: { reasoning: true },
	load: {
		maxSeqLength: 65536,
		cacheTypeKv: "q8_0",
		speculativeType: "mtp",
		specDraftNMax: 4,
		nParallel: 2,
		extraArgs: ["--flash-attn", "--kv-unified"],
	},
	sampling: { temperature: 0.3, topP: 0.9, topK: 40, minP: 0.01, repeatPenalty: 1.1, seed: 42 },
	samplingThinking: { temperature: 0.55, topP: 0.92 },
};

await test("reconfigure: all-empty answers keep every current setting", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", "", "", ""],
		selects: ["__keep", "__keep", "__keep", "__keep", "__keep"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4", contextWindow: 65536 }, detected, existingSettings);
	assert.equal(s.load?.maxSeqLength, 65536);
	assert.equal(s.load?.cacheTypeKv, "q8_0");
	assert.deepEqual(s.load?.speculation, {
		draft: { kind: "mtp", depth: 4 },
		ngram: { kind: "none" },
	});
	assert.equal(s.load?.nParallel, 2);
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn", "--kv-unified"]);
	assert.equal(s.sampling?.temperature, 0.3);
	assert.equal(s.sampling?.seed, 42);
	assert.deepEqual(s.samplingThinking, { temperature: 0.55, topP: 0.92 });
});

await test("reconfigure: '-' clears a value, new value replaces", async () => {
	const { ui } = scripted({
		texts: ["131072", "-", "-", "-", "", "", "", "", "", "", "", "-"], // ctx=131072, clear draft/parallel/extra, sampling defaults, clear seed
		selects: ["f16", "mtp", "none", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected, existingSettings);
	assert.equal(s.load?.maxSeqLength, 131072); // replaced
	assert.equal(s.load?.cacheTypeKv, "f16"); // replaced via select
	assert.deepEqual(s.load?.speculation, {
		draft: { kind: "mtp" },
		ngram: { kind: "none" },
	}); // draft depth cleared
	assert.equal(s.load?.nParallel, undefined); // cleared
	assert.equal(s.load?.extraArgs, undefined); // cleared
	assert.equal(s.sampling?.temperature, 0.3); // kept
	assert.equal(s.sampling?.seed, undefined); // cleared
	assert.equal(s.samplingThinking, undefined); // 'same' clears thinking sampling
});

console.log(`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`);
