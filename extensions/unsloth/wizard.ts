/**
 * Settings wizards for Unsloth models. Runs on a minimal prompt surface so
 * the same code drives pi's /login flow (ProviderAuthInteraction) and the
 * /unsloth command (ctx.ui). No pi imports — unit-testable (test-wizard.ts).
 */

import { createHash } from "node:crypto";
import type { ProviderAuthInteraction } from "@earendil-works/pi-ai";
import type {
	DraftSpeculation,
	LoadSettings,
	NgramSpeculation,
	SamplingSettings,
	SpeculativeSettings,
	UnslothModelSettings,
} from "./config.ts";
import type { DiscoveredModel } from "./discover.ts";
import { matchesActiveModel, thinkingConfigFor, type ReasoningInfo } from "./thinking.ts";
import { fetchHuggingFaceTemplates, mergeTemplateLibrary, type ChatTemplate } from "./chat-template.ts";

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

export function parseIntInRange(
	raw: string | undefined,
	min: number,
	max: number,
	label: string,
): number | undefined {
	const value = parseInt_(raw);
	if (value !== undefined && (value < min || value > max)) {
		throw new Error(`${label} must be between ${min} and ${max}`);
	}
	return value;
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

export async function chatTemplateWizard(
	ui: WizardInteraction,
	current: ChatTemplate | undefined,
	modelHistory: ChatTemplate[],
	allTemplates: ChatTemplate[],
	fetchFromHuggingFace: (model: string) => Promise<ChatTemplate[]> = fetchHuggingFaceTemplates,
): Promise<{ selected?: ChatTemplate; history: ChatTemplate[] }> {
	const own = mergeTemplateLibrary(modelHistory, current ? [current] : []);
	const options = [
		{ id: "__default", label: "Model default (no replacement)" },
		...(current ? [{ id: current.id, label: `Current: ${current.name}` }] : []),
		...own.filter((t) => t.id !== current?.id).map((t) => ({ id: t.id, label: t.name })),
		{ id: "__browse", label: "Browse all saved templates…" },
		{ id: "__huggingface", label: "Get from Hugging Face…" },
		{ id: "__paste", label: "Paste a template directly…" },
	];
	const choice = await ui.select("Chat template", options);
	if (!choice) throw new Error("Cancelled");
	if (choice === "__default") return { selected: undefined, history: own };
	const direct = own.find((t) => t.id === choice);
	if (direct) return { selected: direct, history: own };

	if (choice === "__browse") {
		const library = mergeTemplateLibrary(allTemplates);
		if (library.length === 0) throw new Error("No saved templates yet");
		const picked = await ui.select("Saved templates", library.map((t) => ({ id: t.id, label: t.name })));
		if (!picked) throw new Error("Cancelled");
		const selected = library.find((t) => t.id === picked);
		if (!selected) throw new Error("Template not found");
		return { selected, history: mergeTemplateLibrary(own, [selected]) };
	}

	if (choice === "__huggingface") {
		const model = await ui.text("Hugging Face model or page address", "owner/model");
		ui.progress("Reading chat templates from Hugging Face…");
		const fetched = await fetchFromHuggingFace(model);
		const defaultTemplate = fetched.find((t) => t.isDefault) ?? fetched[0];
		let selected = defaultTemplate;
		if (fetched.length > 1) {
			const picked = await ui.select(
				"Template from Hugging Face (main template shown first)",
				fetched.map((t) => ({ id: t.id, label: `${t.isDefault ? "Main — " : ""}${t.name}` })),
			);
			if (!picked) throw new Error("Cancelled");
			selected = fetched.find((t) => t.id === picked) ?? defaultTemplate;
		}
		return { selected, history: mergeTemplateLibrary(own, fetched) };
	}

	const name = (await ui.text("Template name", "e.g. Qwen tools template")).trim();
	const content = (await ui.text("Paste the complete Jinja chat template")).trim();
	if (!name) throw new Error("Template name is required");
	if (!content) throw new Error("Chat template cannot be empty");
	const id = `pasted:${createHash("sha256").update(content).digest("hex").slice(0, 16)}`;
	const selected: ChatTemplate = { id, name, content };
	return { selected, history: mergeTemplateLibrary(own, [selected]) };
}

const KV_TYPES = ["server default", "f16", "bf16", "q8_0", "q4_0", "iq4_nl"];
const DRAFT_TYPES = [
	{ id: "server-default", label: "Auto — let Unsloth choose" },
	{ id: "none", label: "Off — no draft model" },
	{ id: "mtp", label: "MTP — use prediction heads in the main model" },
	{ id: "dflash", label: "DFlash — use a separate DFlash helper model" },
];
const NGRAM_TYPES = [
	{ id: "none", label: "Off — no n-gram helper" },
	{ id: "cache", label: "Cache — remember repeated text across requests" },
	{ id: "mod", label: "Mod — shared rolling pattern memory" },
	{ id: "simple", label: "Simple — search the current text" },
	{ id: "map-k", label: "Map — indexed pattern search" },
	{ id: "map-k4v", label: "Map-4 — indexed search with four suggestions" },
];
const KV_UNIFIED_FLAGS = new Set(["--kv-unified", "-kvu"]);

function argValue(args: string[] | undefined, names: string[]): string | undefined {
	if (!args) return undefined;
	for (let i = 0; i < args.length - 1; i++) {
		if (names.includes(args[i])) return args[i + 1];
	}
	return undefined;
}

function existingSpeculation(load: LoadSettings | undefined): SpeculativeSettings {
	if (load?.speculation) return load.speculation;
	const rawType = argValue(load?.extraArgs, ["--spec-type"]);
	const combinedType = rawType ?? load?.speculativeType ?? "server-default";
	let draft: DraftSpeculation;
	if (combinedType.includes("dflash")) {
		const hf = argValue(load?.extraArgs, ["--spec-draft-hf", "--hf-repo-draft", "-hfd", "-hfrd"]);
		const file = argValue(load?.extraArgs, ["--spec-draft-model", "--model-draft", "-md"]);
		const source = hf
			? { kind: "huggingface" as const, model: hf }
			: { kind: "file" as const, path: file ?? "" };
		draft = { kind: "dflash", source, ...(load?.specDraftNMax ? { depth: load.specDraftNMax } : {}) };
	} else if (combinedType.includes("mtp")) {
		draft = { kind: "mtp", ...(load?.specDraftNMax ? { depth: load.specDraftNMax } : {}) };
	} else if (combinedType === "off" || combinedType === "none" || combinedType.startsWith("ngram")) {
		draft = { kind: "none" };
	} else {
		draft = { kind: "server-default" };
	}

	const ngramName = ["cache", "mod", "simple", "map-k", "map-k4v"].find((kind) => combinedType.includes(`ngram-${kind}`));
	const ngram: NgramSpeculation = ngramName === "cache"
		? { kind: "cache" }
		: ngramName === "mod" || combinedType === "ngram" || combinedType === "mtp+ngram"
			? { kind: "mod" }
			: ngramName === "simple" || ngramName === "map-k" || ngramName === "map-k4v"
				? { kind: ngramName }
				: { kind: "none" };
	return { draft, ngram };
}

async function speculationWizard(ui: WizardInteraction, existing?: LoadSettings): Promise<SpeculativeSettings> {
	const current = existingSpeculation(existing);
	const draftOptions = [
		...(existing ? [{ id: "__keep", label: `Keep current (${current.draft.kind})` }] : []),
		...DRAFT_TYPES,
	];
	const draftChoice = await ui.select("Draft method", draftOptions);
	if (!draftChoice) throw new Error("Cancelled");

	const depthRaw = (
		await ui.text(
			"Draft depth (1–16)",
			"depth" in current.draft && current.draft.depth ? `current: ${current.draft.depth}` : "server default",
		)
	).trim();
	const depth = depthRaw === "-"
		? undefined
		: (parseIntInRange(depthRaw, 1, 16, "Draft depth") ?? ("depth" in current.draft ? current.draft.depth : undefined));

	let draft: DraftSpeculation;
	if (draftChoice === "__keep") {
		draft = current.draft.kind === "mtp"
			? { kind: "mtp", ...(depth ? { depth } : {}) }
			: current.draft.kind === "dflash"
				? { kind: "dflash", source: current.draft.source, ...(depth ? { depth } : {}) }
				: current.draft;
	} else if (draftChoice === "mtp") {
		draft = { kind: "mtp", ...(depth ? { depth } : {}) };
	} else if (draftChoice === "dflash") {
		const currentSource = current.draft.kind === "dflash" ? current.draft.source : undefined;
		const sourceChoice = await ui.select("DFlash helper location", [
			...(currentSource ? [{ id: "__keep", label: `Keep current (${currentSource.kind})` }] : []),
			{ id: "huggingface", label: "Hugging Face model and quant" },
			{ id: "file", label: "GGUF file already on the Unsloth computer" },
		]);
		if (!sourceChoice) throw new Error("Cancelled");
		const sourceKind = sourceChoice === "__keep" ? currentSource?.kind : sourceChoice;
		if (sourceKind === "huggingface") {
			const currentModel = currentSource?.kind === "huggingface" ? currentSource.model : "";
			const model = (await ui.text("DFlash Hugging Face model", currentModel || "owner/model:quant")).trim() || currentModel;
			if (!model) throw new Error("A DFlash helper model is required");
			draft = { kind: "dflash", source: { kind: "huggingface", model }, ...(depth ? { depth } : {}) };
		} else {
			const currentPath = currentSource?.kind === "file" ? currentSource.path : "";
			const path = (await ui.text("DFlash GGUF file on the Unsloth computer", currentPath || "C:\\models\\draft.gguf")).trim() || currentPath;
			if (!path) throw new Error("A DFlash helper file is required");
			draft = { kind: "dflash", source: { kind: "file", path }, ...(depth ? { depth } : {}) };
		}
	} else {
		draft = draftChoice === "none" ? { kind: "none" } : { kind: "server-default" };
	}

	const ngramOptions = [
		...(existing ? [{ id: "__keep", label: `Keep current (${current.ngram.kind})` }] : []),
		...NGRAM_TYPES,
	];
	const ngramChoice = await ui.select("N-gram helper (can run beside the draft method)", ngramOptions);
	if (!ngramChoice) throw new Error("Cancelled");
	if (ngramChoice === "__keep") return { draft, ngram: current.ngram };
	if (ngramChoice === "none" || ngramChoice === "cache") return { draft, ngram: { kind: ngramChoice } };

	if (ngramChoice === "mod") {
		const match = parseInt_(await ui.text("N-gram lookup length", "24 (server default)"));
		const min = parseInt_(await ui.text("Minimum n-gram draft length", "48 (server default)"));
		const max = parseInt_(await ui.text("Maximum n-gram draft length", "64 (server default)"));
		return { draft, ngram: { kind: "mod", ...(match ? { match } : {}), ...(min ? { min } : {}), ...(max ? { max } : {}) } };
	}

	const sizeN = parseInt_(await ui.text("N-gram lookup length", "12 (server default)"));
	const sizeM = parseInt_(await ui.text("N-gram proposed length", "48 (server default)"));
	const minHits = parseInt_(await ui.text("Matches required before use", "1 (server default)"));
	return {
		draft,
		ngram: {
			kind: ngramChoice,
			...(sizeN ? { sizeN } : {}),
			...(sizeM ? { sizeM } : {}),
			...(minHits ? { minHits } : {}),
		},
	};
}

function setKvUnified(args: string[] | undefined, enabled: boolean): string[] | undefined {
	const filtered = (args ?? []).filter((arg) => !KV_UNIFIED_FLAGS.has(arg));
	if (enabled) filtered.push("--kv-unified");
	return filtered.length > 0 ? filtered : undefined;
}

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

	const speculation = await speculationWizard(ui, existing?.load);

	const parallelRaw = (
		await ui.text(
			"Parallel decode slots",
			existing?.load?.nParallel ? `current: ${existing.load.nParallel}` : "1 (default)",
		)
	).trim();
	const nParallel = parallelRaw === "-" ? undefined : (parseInt_(parallelRaw) ?? existing?.load?.nParallel);

	let kvUnified: boolean | undefined;
	if (nParallel !== undefined && nParallel > 1) {
		const existingParallel = existing?.load?.nParallel;
		const existingHasKvUnified = existing?.load?.extraArgs?.some((arg) => KV_UNIFIED_FLAGS.has(arg)) ?? false;
		const kvOptions = [
			...(existingParallel !== undefined && existingParallel > 1
				? [{ id: "__keep", label: `Keep current (${existingHasKvUnified ? "enabled" : "off"})` }]
				: []),
			{ id: "yes", label: "Yes — share the context pool (recommended)" },
			{ id: "no", label: "No — separate context per slot" },
		];
		const kvChoice = await ui.select("Enable shared context for parallel slots?", kvOptions);
		if (!kvChoice) throw new Error("Cancelled");
		kvUnified = kvChoice === "__keep" ? existingHasKvUnified : kvChoice === "yes";
	}

	const extraRaw = (
		await ui.text(
			"Extra llama.cpp args",
			existing?.load?.extraArgs?.length
				? `current: ${existing.load.extraArgs.join(" ")}`
				: "e.g. --n-cpu-moe 4 --flash-attn",
		)
	).trim();
	const enteredExtraArgs = extraRaw === "-" ? undefined : extraRaw === "" ? existing?.load?.extraArgs : extraRaw.split(/\s+/);
	const extraArgs =
		nParallel === 1
			? setKvUnified(enteredExtraArgs, false)
			: kvUnified === undefined
				? enteredExtraArgs
				: setKvUnified(enteredExtraArgs, kvUnified);

	const load: LoadSettings = {
		...(maxSeqLength ? { maxSeqLength } : {}),
		...(cacheTypeKv ? { cacheTypeKv } : {}),
		speculation,
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
