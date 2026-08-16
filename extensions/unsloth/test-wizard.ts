/**
 * Unit tests for wizard.ts — run with: node test-wizard.ts
 */
import assert from "node:assert/strict";
import {
	parseNumber,
	parseInt_,
	normalizeBaseUrl,
	providerSetup,
	pickModel,
	settingsWizard,
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
	const ui: WizardInteraction = {
		async text(message) {
			const next = texts.shift();
			if (next === undefined) throw new Error(`unexpected text prompt: ${message}`);
			return next;
		},
		async select(message) {
			if (selects.length === 0) throw new Error(`unexpected select: ${message}`);
			return selects.shift();
		},
		progress(m) {
			progress.push(m);
		},
	};
	return { ui, progress };
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

const detected: ReasoningInfo = {
	style: "enable_thinking_effort",
	levels: ["high", "max"],
	supportsPreserveThinking: false,
	activeModel: "org/m",
};

await test("settingsWizard: full answers flow into all settings groups", async () => {
	const { ui, progress } = scripted({
		texts: [
			"65536", // context
			"4",     // spec draft n
			"2",     // parallel
			"--flash-attn", // extra args
			"0.7", "0.8", "20", "0.05", "1.1", "", // sampling (temp, top_p, top_k, min_p, rep, seed)
			"0.6", "0.95", "20", "0.0", "1.0", "", // thinking sampling (custom)
		],
		selects: ["q8_0", "off", "custom"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected);
	assert.equal(s.contextWindow, 65536);
	assert.equal(s.load?.cacheTypeKv, "q8_0");
	assert.equal(s.load?.speculativeType, "off");
	assert.equal(s.load?.specDraftNMax, 4);
	assert.equal(s.load?.nParallel, 2);
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn"]);
	assert.equal(s.sampling?.temperature, 0.7);
	assert.equal(s.sampling?.minP, 0.05);
	assert.equal(s.samplingThinking?.temperature, 0.6);
	assert.equal(s.samplingThinking?.topP, 0.95);
	// thinking auto-configured from detection (loaded model)
	assert.equal(s.thinking?.reasoning, true);
	assert.equal(s.thinking?.thinkingLevelMap?.max, "max");
	assert.ok(progress.some((p) => p.includes("enable_thinking_effort")));
});

await test("settingsWizard: empty answers keep defaults, 'same' skips thinking sampling", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "auto", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected);
	assert.equal(s.contextWindow, undefined); // model carried none
	assert.deepEqual(s.load, {});
	assert.equal(s.sampling?.temperature, 0.7); // default
	assert.equal(s.samplingThinking, undefined);
});

await test("settingsWizard: non-loaded model without detection gets reasoning=false, no thinking question", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "auto"],
	});
	const s = await settingsWizard(ui, { id: "org/other:Q4" }, detected);
	assert.deepEqual(s.thinking, { reasoning: false });
});

await test("settingsWizard: 'recommended' thinking sampling preset", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", ""],
		selects: ["server default", "auto", "recommended"],
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
		speculativeType: "off",
		specDraftNMax: 4,
		nParallel: 2,
		extraArgs: ["--flash-attn"],
	},
	sampling: { temperature: 0.3, topP: 0.9, topK: 40, minP: 0.01, repeatPenalty: 1.1, seed: 42 },
	samplingThinking: { temperature: 0.55, topP: 0.92 },
};

await test("reconfigure: all-empty answers keep every current setting", async () => {
	const { ui } = scripted({
		texts: ["", "", "", "", "", "", "", "", "", ""],
		selects: ["__keep", "__keep", "__keep"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4", contextWindow: 65536 }, detected, existingSettings);
	assert.equal(s.load?.maxSeqLength, 65536);
	assert.equal(s.load?.cacheTypeKv, "q8_0");
	assert.equal(s.load?.speculativeType, "off");
	assert.equal(s.load?.specDraftNMax, 4);
	assert.equal(s.load?.nParallel, 2);
	assert.deepEqual(s.load?.extraArgs, ["--flash-attn"]);
	assert.equal(s.sampling?.temperature, 0.3);
	assert.equal(s.sampling?.seed, 42);
	assert.deepEqual(s.samplingThinking, { temperature: 0.55, topP: 0.92 });
});

await test("reconfigure: '-' clears a value, new value replaces", async () => {
	const { ui } = scripted({
		texts: ["131072", "-", "-", "-", "", "", "", "", "", "-"], // ctx=131072, clear draft/parallel/extra, sampling defaults, clear seed
		selects: ["f16", "mtp", "same"],
	});
	const s = await settingsWizard(ui, { id: "org/m:Q4" }, detected, existingSettings);
	assert.equal(s.load?.maxSeqLength, 131072); // replaced
	assert.equal(s.load?.cacheTypeKv, "f16"); // replaced via select
	assert.equal(s.load?.speculativeType, "mtp");
	assert.equal(s.load?.specDraftNMax, undefined); // cleared
	assert.equal(s.load?.nParallel, undefined); // cleared
	assert.equal(s.load?.extraArgs, undefined); // cleared
	assert.equal(s.sampling?.temperature, 0.3); // kept
	assert.equal(s.sampling?.seed, undefined); // cleared
	assert.equal(s.samplingThinking, undefined); // 'same' clears thinking sampling
});

console.log(`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`);
