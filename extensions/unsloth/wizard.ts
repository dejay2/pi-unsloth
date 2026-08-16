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
	ui.progress("Load-time settings (applied when the model loads — Enter keeps server defaults)");
	const maxSeqLength = parseInt_(
		await ui.text(
			"Context size (tokens)",
			existing?.load?.maxSeqLength?.toString() ??
				(model.contextWindow ? `native: ${model.contextWindow}` : "server default"),
		),
	);
	const kvChoice = await ui.select(
		"KV cache dtype",
		KV_TYPES.map((k) => ({ id: k, label: k })),
	);
	if (!kvChoice) throw new Error("Cancelled");
	const specChoice = await ui.select(
		"Speculative decoding (MTP)",
		SPEC_TYPES.map((s) => ({ id: s.split(" ")[0], label: s })),
	);
	if (!specChoice) throw new Error("Cancelled");
	const specDraftNMax = parseInt_(await ui.text("Speculative draft tokens (1–16)", "auto"));
	const nParallel = parseInt_(await ui.text("Parallel decode slots", "1 (default)"));
	const extraRaw = await ui.text("Extra llama.cpp args (optional)", "e.g. --n-cpu-moe 4 --flash-attn");
	const extraArgs = extraRaw.trim() ? extraRaw.trim().split(/\s+/) : undefined;

	const load: LoadSettings = {
		...(maxSeqLength ? { maxSeqLength } : {}),
		...(kvChoice !== "server default" ? { cacheTypeKv: kvChoice } : {}),
		...(specChoice !== "auto" ? { speculativeType: specChoice } : {}),
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
		repeatPenalty: 1.05,
	}, existing?.sampling);

	// --- thinking-conditional sampling ---
	let samplingThinking: SamplingSettings | undefined;
	if (thinking.reasoning && !alwaysOn) {
		const choice = await ui.select("Different sampling while thinking?", [
			{ id: "same", label: "Same as normal" },
			{ id: "recommended", label: "Qwen recommended (temp 0.6, top_p 0.95)" },
			{ id: "custom", label: "Custom…" },
		]);
		if (!choice) throw new Error("Cancelled");
		if (choice === "recommended") {
			samplingThinking = { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 };
		} else if (choice === "custom") {
			samplingThinking = await samplingWizard(ui, { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }, existing?.samplingThinking);
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
	const temperature = await ui.text("Temperature", String(cur.temperature));
	const topP = await ui.text("top_p", String(cur.topP));
	const topK = await ui.text("top_k", String(cur.topK));
	const minP = await ui.text("min_p", String(cur.minP));
	const repeatPenalty = await ui.text("Repetition penalty", String(cur.repeatPenalty));
	const seed = await ui.text("Seed (optional)", "random");
	return {
		temperature: parseNumber(temperature) ?? cur.temperature,
		topP: parseNumber(topP) ?? cur.topP,
		topK: parseInt_(topK) ?? cur.topK,
		minP: parseNumber(minP) ?? cur.minP,
		repeatPenalty: parseNumber(repeatPenalty) ?? cur.repeatPenalty,
		...(parseInt_(seed) !== undefined ? { seed: parseInt_(seed) } : {}),
	};
}
