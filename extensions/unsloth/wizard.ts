/**
 * Settings wizards for Unsloth models. Runs on a minimal prompt surface so
 * the same code drives pi's /login flow (ProviderAuthInteraction) and the
 * /unsloth command (ctx.ui). No pi imports — unit-testable (test-wizard.ts).
 */

import type { ProviderAuthInteraction } from "@earendil-works/pi-ai";
import type { LoadSettings, SamplingSettings, UnslothModelSettings } from "./config.ts";
import type { DiscoveredModel } from "./discover.ts";
import { matchesActiveModel, thinkingConfigFor, type ReasoningInfo } from "./thinking.ts";

/** Minimal prompt surface — implemented by pi's login UI or ctx.ui. */
export interface WizardInteraction {
	text(message: string, placeholder?: string, secret?: boolean): Promise<string>;
	select(message: string, options: { id: string; label: string }[]): Promise<string | undefined>;
	progress(message: string): void;
}

/** Adapt pi's ProviderAuthInteraction (used by both api-key and OAuth logins). */
export function adaptAuthInteraction(i: ProviderAuthInteraction): WizardInteraction {
	return {
		text: (message, placeholder, secret) =>
			i.prompt({ type: secret ? "secret" : "text", message, placeholder }),
		select: (message, options) => i.prompt({ type: "select", message, options }),
		progress: (message) => i.notify({ type: "progress", message }),
	};
}

// ---------------------------------------------------------------------------
// Parsing helpers (exported for tests)
// ---------------------------------------------------------------------------

export function parseNumber(raw: string | undefined): number | undefined {
	if (!raw || !raw.trim()) return undefined;
	const n = Number(raw.trim());
	if (!Number.isFinite(n) || n < 0) throw new Error(`"${raw.trim()}" is not a valid number`);
	return n;
}

export function parseInt_(raw: string | undefined): number | undefined {
	const n = parseNumber(raw);
	if (n === undefined) return undefined;
	if (!Number.isInteger(n)) throw new Error(`"${raw}" is not a whole number`);
	return n;
}

/** Normalize a server URL to pi's baseUrl form (ends in /v1). */
export function normalizeBaseUrl(raw: string): string {
	let url = raw.trim().replace(/\/+$/, "");
	if (!/^https?:\/\//.test(url)) url = `http://${url}`;
	if (!/\/v1$/.test(url)) url = `${url}/v1`;
	return url;
}

// ---------------------------------------------------------------------------
// Wizard steps
// ---------------------------------------------------------------------------

/** Step: server URL + API key. */
export async function providerSetup(
	ui: WizardInteraction,
	defaults?: { baseUrl?: string },
): Promise<{ baseUrl: string; apiKey: string }> {
	const rawUrl = await ui.text("Unsloth server URL", defaults?.baseUrl ?? "http://localhost:8888");
	const baseUrl = normalizeBaseUrl(rawUrl);
	const apiKey = (await ui.text("Unsloth API key", "sk-unsloth-...", true)).trim();
	if (!apiKey) throw new Error("API key is required");
	return { baseUrl, apiKey };
}

/** Step: pick one model (login surface is single-select). */
export async function pickModel(ui: WizardInteraction, models: DiscoveredModel[]): Promise<DiscoveredModel> {
	const choice = await ui.select(
		"Which model?",
		models.map((m) => ({ id: m.id, label: m.displayName ?? m.id })),
	);
	if (!choice) throw new Error("Cancelled");
	const model = models.find((m) => m.id === choice);
	if (!model) throw new Error("Model not found");
	return model;
}

const KV_TYPES = ["server default", "f16", "bf16", "q8_0", "q4_0", "iq4_nl"];
const SPEC_TYPES = [
	"auto — MTP when the GGUF supports it (recommended)",
	"off — no speculative decoding",
	"mtp — force MTP draft",
	"mtp+ngram — MTP with ngram fallback",
	"ngram — ngram fallback only",
];

/**
 * Step: full per-model settings. Load-time (llama.cpp structural), sampling,
 * and thinking-conditional sampling. Empty answers keep server defaults.
 */
export async function settingsWizard(
	ui: WizardInteraction,
	model: DiscoveredModel,
	detected: ReasoningInfo | null,
	existing?: UnslothModelSettings,
): Promise<UnslothModelSettings> {
	const isLoaded = detected !== null && matchesActiveModel(model.id, detected.activeModel);
	const alwaysOn = isLoaded && detected.style === "always_on";
	const thinking = isLoaded
		? thinkingConfigFor(detected)
		: (existing?.thinking ?? { reasoning: false });
	if (isLoaded && detected.style !== "none") {
		ui.progress(
			`Thinking: ${detected.style}${detected.levels.length ? ` (${detected.levels.join("/")})` : ""} — auto-configured.`,
		);
	}

	// --- load-time settings ---
	ui.progress("Load-time settings (applied when the model loads — Enter keeps current, '-' clears)");
	const ctxRaw = (
		await ui.text(
			"Context size (tokens)",
			existing?.load?.maxSeqLength
				? `current: ${existing.load.maxSeqLength}`
				: model.contextWindow
					? `native: ${model.contextWindow}`
					: "server default",
		)
	).trim();
	const maxSeqLength = ctxRaw === "-" ? undefined : (parseInt_(ctxRaw) ?? existing?.load?.maxSeqLength);

	const kvOptions = KV_TYPES.map((k) => ({ id: k, label: k }));
	if (existing?.load?.cacheTypeKv) {
		kvOptions.unshift({ id: "__keep", label: `Keep current (${existing.load.cacheTypeKv})` });
	}
	const kvChoice = await ui.select("KV cache dtype", kvOptions);
	if (!kvChoice) throw new Error("Cancelled");
	const cacheTypeKv =
		kvChoice === "__keep" ? existing?.load?.cacheTypeKv : kvChoice !== "server default" ? kvChoice : undefined;

	const specOptions = SPEC_TYPES.map((s) => ({ id: s.split(" ")[0], label: s }));
	if (existing?.load?.speculativeType) {
		specOptions.unshift({ id: "__keep", label: `Keep current (${existing.load.speculativeType})` });
	}
	const specChoice = await ui.select("Speculative decoding (MTP)", specOptions);
	if (!specChoice) throw new Error("Cancelled");
	const speculativeType =
		specChoice === "__keep" ? existing?.load?.speculativeType : specChoice !== "auto" ? specChoice : undefined;

	const draftRaw = (
		await ui.text(
			"Speculative draft tokens (1–16)",
			existing?.load?.specDraftNMax ? `current: ${existing.load.specDraftNMax}` : "auto",
		)
	).trim();
	const specDraftNMax = draftRaw === "-" ? undefined : (parseInt_(draftRaw) ?? existing?.load?.specDraftNMax);

	const parallelRaw = (
		await ui.text(
			"Parallel decode slots",
			existing?.load?.nParallel ? `current: ${existing.load.nParallel}` : "1 (default)",
		)
	).trim();
	const nParallel = parallelRaw === "-" ? undefined : (parseInt_(parallelRaw) ?? existing?.load?.nParallel);

	const extraRaw = (
		await ui.text(
			"Extra llama.cpp args",
			existing?.load?.extraArgs?.length
				? `current: ${existing.load.extraArgs.join(" ")}`
				: "e.g. --n-cpu-moe 4 --flash-attn",
		)
	).trim();
	const extraArgs = extraRaw === "-" ? undefined : extraRaw === "" ? existing?.load?.extraArgs : extraRaw.split(/\s+/);

	const load: LoadSettings = {
		...(maxSeqLength ? { maxSeqLength } : {}),
		...(cacheTypeKv ? { cacheTypeKv } : {}),
		...(speculativeType ? { speculativeType } : {}),
		...(specDraftNMax ? { specDraftNMax } : {}),
		...(nParallel ? { nParallel } : {}),
		...(extraArgs ? { extraArgs } : {}),
	};

	// --- sampling (always applied) ---
	ui.progress("Sampling for normal (non-thinking) requests — Enter keeps the shown default");
	const sampling = await samplingWizard(ui, {
		temperature: 0.7,
		topP: 0.8,
		topK: 20,
		minP: 0.0,
		presencePenalty: 0.0,
		frequencyPenalty: 0.0,
		repeatPenalty: 1.05,
	}, existing?.sampling);

	// --- thinking-conditional sampling ---
	let samplingThinking: SamplingSettings | undefined;
	if (thinking.reasoning && !alwaysOn) {
		const thinkOptions = [
			...(existing?.samplingThinking
				? [
						{
							id: "__keep",
							label: `Keep current (temp ${existing.samplingThinking.temperature ?? "—"}, top_p ${existing.samplingThinking.topP ?? "—"}, …)`,
						},
					]
				: []),
			{ id: "same", label: "Same as normal" },
			{ id: "recommended", label: "Qwen recommended (temp 0.6, top_p 0.95)" },
			{ id: "custom", label: "Custom…" },
		];
		const choice = await ui.select("Different sampling while thinking?", thinkOptions);
		if (!choice) throw new Error("Cancelled");
		if (choice === "__keep") {
			samplingThinking = existing?.samplingThinking;
		} else if (choice === "recommended") {
			samplingThinking = { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 };
		} else if (choice === "custom") {
			samplingThinking = await samplingWizard(ui, { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0, presencePenalty: 0.0, frequencyPenalty: 0.0, repeatPenalty: 1.0 }, existing?.samplingThinking);
		}
	}

	return {
		...(maxSeqLength ? { contextWindow: maxSeqLength } : model.contextWindow ? { contextWindow: model.contextWindow } : {}),
		...(model.maxTokens ? { maxTokens: model.maxTokens } : {}),
		thinking,
		load,
		sampling,
		...(samplingThinking ? { samplingThinking } : {}),
	};
}

async function samplingWizard(
	ui: WizardInteraction,
	defaults: Required<Omit<SamplingSettings, "seed">> & { seed?: number },
	existing?: SamplingSettings,
): Promise<SamplingSettings> {
	const cur = { ...defaults, ...existing };
	const temperature = await ui.text("Temperature (randomness — lower = more deterministic)", String(cur.temperature));
	const topP = await ui.text("top_p (nucleus sampling)", String(cur.topP));
	const topK = await ui.text("top_k (candidate pool size)", String(cur.topK));
	const minP = await ui.text("min_p (minimum token probability)", String(cur.minP));
	const presencePenalty = await ui.text("Presence penalty (discourages repeating topics)", String(cur.presencePenalty));
	const frequencyPenalty = await ui.text("Frequency penalty (discourages repeating tokens)", String(cur.frequencyPenalty));
	const repeatPenalty = await ui.text("Repetition penalty (llama.cpp repeat_penalty)", String(cur.repeatPenalty));
	const seed = await ui.text(
		"Seed (pins the RNG for reproducible outputs)",
		existing?.seed !== undefined ? `current: ${existing.seed}  (Enter = keep, - = clear)` : "random",
	);
	const seedTrimmed = seed.trim();
	return {
		temperature: parseNumber(temperature) ?? cur.temperature,
		topP: parseNumber(topP) ?? cur.topP,
		topK: parseInt_(topK) ?? cur.topK,
		minP: parseNumber(minP) ?? cur.minP,
		presencePenalty: parseNumber(presencePenalty) ?? cur.presencePenalty,
		frequencyPenalty: parseNumber(frequencyPenalty) ?? cur.frequencyPenalty,
		repeatPenalty: parseNumber(repeatPenalty) ?? cur.repeatPenalty,
		...(seedTrimmed === "-" ? {} : parseInt_(seedTrimmed) !== undefined ? { seed: parseInt_(seedTrimmed) } : cur.seed !== undefined ? { seed: cur.seed } : {}),
	};
}
