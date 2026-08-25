/**
 * Per-provider/model settings store for Unsloth servers, plus the pure
 * payload builders that turn settings into Unsloth/pi wire formats.
 *
 * Lives at ~/.pi/agent/unsloth.json. No pi imports — unit-testable.
 */

import {
	existsSync,
	mkdirSync,
	readFileSync,
	renameSync,
	writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import type { ThinkingConfig } from "./thinking.ts";
import { mergeTemplateLibrary, type ChatTemplate } from "./chat-template.ts";

export const CONFIG_PATH = join(homedir(), ".pi", "agent", "unsloth.json");

export interface SamplingSettings {
	temperature?: number;
	topP?: number;
	topK?: number;
	minP?: number;
	presencePenalty?: number;
	frequencyPenalty?: number;
	repeatPenalty?: number;
	seed?: number;
}

export type DraftModelSource =
	| { kind: "huggingface"; model: string }
	| { kind: "file"; path: string };

export type DraftSpeculation =
	| { kind: "server-default" }
	| { kind: "none" }
	| { kind: "mtp"; depth?: number }
	| { kind: "dflash"; source: DraftModelSource; depth?: number };

export type NgramSpeculation =
	| { kind: "none" }
	| { kind: "cache" }
	| { kind: "mod"; match?: number; min?: number; max?: number }
	| {
			kind: "simple" | "map-k" | "map-k4v";
			sizeN?: number;
			sizeM?: number;
			minHits?: number;
	  };

export interface SpeculativeSettings {
	draft: DraftSpeculation;
	ngram: NgramSpeculation;
}

/** Load-time (structural) llama.cpp settings, applied via POST /api/inference/load. */
export interface LoadSettings {
	maxSeqLength?: number;
	cacheTypeKv?: string;
	/** Structured speculative decoding settings. Preferred over the legacy fields below. */
	speculation?: SpeculativeSettings;
	/** Legacy settings retained so existing saved files keep working. */
	speculativeType?: string;
	specDraftNMax?: number;
	nParallel?: number;
	nBatch?: number;
	nUBatch?: number;
	extraArgs?: string[];
}

export interface UnslothModelSettings {
	contextWindow?: number;
	/** Active replacement. Undefined means use the template built into the model. */
	chatTemplate?: ChatTemplate;
	/** Templates previously selected for this model, offered on reconfiguration. */
	chatTemplates?: ChatTemplate[];
	maxTokens?: number;
	thinking?: ThinkingConfig;
	load?: LoadSettings;
	/** Applied always (written to models.json samplingParams). */
	sampling?: SamplingSettings;
	/** Applied on top when pi's thinking level is not "off" (payload hook). */
	samplingThinking?: SamplingSettings;
}

export interface UnslothProviderConfig {
	/** Base URL including /v1 (what pi uses). */
	baseUrl: string;
	/** Literal key or $ENV reference. */
	apiKey: string;
	/** Model remembered by Unsloth Studio. Never chooses Pi's startup model. */
	defaultModelId?: string;
	models: Record<string, UnslothModelSettings>;
}

export interface UnslothConfig {
	providers: Record<string, UnslothProviderConfig>;
}

export function readConfig(path: string = CONFIG_PATH): UnslothConfig {
	if (!existsSync(path)) return { providers: {} };
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8"));
		if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
			return { providers: {} };
		if (!parsed.providers || typeof parsed.providers !== "object")
			parsed.providers = {};
		return parsed as UnslothConfig;
	} catch {
		return { providers: {} };
	}
}

export function writeConfig(
	data: UnslothConfig,
	path: string = CONFIG_PATH,
): void {
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.tmp-${process.pid}`;
	writeFileSync(tmp, JSON.stringify(data, null, 2) + "\n", "utf8");
	renameSync(tmp, path);
}

export function upsertModel(
	data: UnslothConfig,
	providerId: string,
	provider: { baseUrl: string; apiKey: string },
	modelId: string,
	settings: UnslothModelSettings,
): void {
	const p = (data.providers[providerId] ??= {
		baseUrl: provider.baseUrl,
		apiKey: provider.apiKey,
		models: {},
	});
	p.baseUrl = provider.baseUrl;
	p.apiKey = provider.apiKey;
	p.models[modelId] = settings;
}

export function findConfiguredModelId(
	provider: UnslothProviderConfig | undefined,
	activeModelId: string | undefined,
): string | undefined {
	if (!provider || !activeModelId) return undefined;
	if (activeModelId in provider.models) return activeModelId;
	return Object.keys(provider.models).find(
		(modelId) => splitModelRef(modelId).modelPath === activeModelId,
	);
}

export function setDefaultModel(
	data: UnslothConfig,
	providerId: string,
	modelId: string,
): boolean {
	const provider = data.providers[providerId];
	if (!provider || !(modelId in provider.models)) return false;
	provider.defaultModelId = modelId;
	return true;
}

export function removeModel(
	data: UnslothConfig,
	providerId: string,
	modelId: string,
): boolean {
	const p = data.providers[providerId];
	if (!p || !(modelId in p.models)) return false;
	delete p.models[modelId];
	if (p.defaultModelId === modelId) delete p.defaultModelId;
	return true;
}

/** Build the reusable template library from every configured model. */
export function collectChatTemplates(data: UnslothConfig): ChatTemplate[] {
	const groups: ChatTemplate[][] = [];
	for (const provider of Object.values(data.providers)) {
		for (const settings of Object.values(provider.models)) {
			groups.push(settings.chatTemplates ?? []);
			if (settings.chatTemplate) groups.push([settings.chatTemplate]);
		}
	}
	return mergeTemplateLibrary(...groups);
}

// ---------------------------------------------------------------------------
// Payload builders
// ---------------------------------------------------------------------------

/** Split "org/repo:QUANT" into the fields Unsloth's /load expects. */
export function splitModelRef(modelId: string): {
	modelPath: string;
	variant?: string;
} {
	const i = modelId.lastIndexOf(":");
	if (i <= 0) return { modelPath: modelId };
	return { modelPath: modelId.slice(0, i), variant: modelId.slice(i + 1) };
}

const MANAGED_SPEC_FLAGS = new Set([
	"--spec-type",
	"--spec-draft-model",
	"--model-draft",
	"-md",
	"--spec-draft-hf",
	"--hf-repo-draft",
	"-hfd",
	"-hfrd",
	"--spec-draft-n-max",
	"--spec-ngram-mod-n-match",
	"--spec-ngram-mod-n-min",
	"--spec-ngram-mod-n-max",
	"--spec-ngram-simple-size-n",
	"--spec-ngram-simple-size-m",
	"--spec-ngram-simple-min-hits",
	"--spec-ngram-map-k-size-n",
	"--spec-ngram-map-k-size-m",
	"--spec-ngram-map-k-min-hits",
	"--spec-ngram-map-k4v-size-n",
	"--spec-ngram-map-k4v-size-m",
	"--spec-ngram-map-k4v-min-hits",
]);

function removeManagedSpecArgs(args: string[] | undefined): string[] {
	const kept: string[] = [];
	for (let i = 0; i < (args?.length ?? 0); i++) {
		const arg = args?.[i];
		if (arg && MANAGED_SPEC_FLAGS.has(arg)) {
			i++;
			continue;
		}
		if (arg && [...MANAGED_SPEC_FLAGS].some((flag) => arg.startsWith(`${flag}=`)))
			continue;
		if (arg !== undefined) kept.push(arg);
	}
	return kept;
}

function appendNumberArg(
	args: string[],
	flag: string,
	value: number | undefined,
): void {
	if (value !== undefined) args.push(flag, String(value));
}

function buildStructuredSpeculation(speculation: SpeculativeSettings): {
	speculativeType?: string;
	draftDepth?: number;
	args: string[];
} {
	const args: string[] = [];
	const types: string[] = [];
	let speculativeType: string | undefined;
	let draftDepth: number | undefined;

	switch (speculation.draft.kind) {
		case "server-default":
			break;
		case "none":
			speculativeType = "off";
			break;
		case "mtp":
			speculativeType = "mtp";
			types.push("draft-mtp");
			draftDepth = speculation.draft.depth;
			break;
		case "dflash":
			speculativeType = "dflash";
			types.push("draft-dflash");
			draftDepth = speculation.draft.depth;
			if (speculation.draft.source.kind === "huggingface") {
				args.push("--spec-draft-hf", speculation.draft.source.model);
			} else {
				args.push("--spec-draft-model", speculation.draft.source.path);
			}
			break;
		default: {
			const exhaustive: never = speculation.draft;
			return exhaustive;
		}
	}

	const ngram = speculation.ngram;
	if (ngram.kind !== "none") {
		types.push(`ngram-${ngram.kind}`);
		if (speculation.draft.kind === "none") speculativeType = "ngram";
	}
	const usesNativeMtpNgram =
		speculation.draft.kind === "mtp" && ngram.kind === "mod";
	if (usesNativeMtpNgram) {
		speculativeType = "mtp+ngram";
	} else {
		if (types.length > 0) args.push("--spec-type", types.join(","));
		else if (speculation.draft.kind === "none") args.push("--spec-type", "none");
		appendNumberArg(args, "--spec-draft-n-max", draftDepth);
	}

	if (ngram.kind === "mod") {
		appendNumberArg(args, "--spec-ngram-mod-n-match", ngram.match);
		appendNumberArg(args, "--spec-ngram-mod-n-min", ngram.min);
		appendNumberArg(args, "--spec-ngram-mod-n-max", ngram.max);
	} else if (
		ngram.kind === "simple" ||
		ngram.kind === "map-k" ||
		ngram.kind === "map-k4v"
	) {
		const prefix = `--spec-ngram-${ngram.kind}`;
		appendNumberArg(args, `${prefix}-size-n`, ngram.sizeN);
		appendNumberArg(args, `${prefix}-size-m`, ngram.sizeM);
		appendNumberArg(args, `${prefix}-min-hits`, ngram.minHits);
	}

	return { speculativeType, draftDepth, args };
}

function resolvedLoadSettings(load: LoadSettings): {
	speculativeType?: string;
	draftDepth?: number;
	extraArgs: string[];
} {
	if (!load.speculation) {
		return {
			speculativeType: load.speculativeType,
			draftDepth: load.specDraftNMax,
			extraArgs: load.extraArgs ?? [],
		};
	}
	const structured = buildStructuredSpeculation(load.speculation);
	return {
		...structured,
		extraArgs: [...removeManagedSpecArgs(load.extraArgs), ...structured.args],
	};
}

/** Body for POST /api/inference/load (== /v1/load). Undefined/empty fields omitted. */
export function buildLoadPayload(
	modelId: string,
	s: UnslothModelSettings,
): Record<string, unknown> {
	const { modelPath, variant } = splitModelRef(modelId);
	const load = s.load ?? {};
	const resolved = resolvedLoadSettings(load);
	const payload: Record<string, unknown> = { model_path: modelPath };
	if (variant) payload.gguf_variant = variant;
	if (s.chatTemplate?.content)
		payload.chat_template_override = s.chatTemplate.content;
	if (load.maxSeqLength) payload.max_seq_length = load.maxSeqLength;
	if (load.cacheTypeKv) payload.cache_type_kv = load.cacheTypeKv;
	if (resolved.speculativeType)
		payload.speculative_type = resolved.speculativeType;
	if (resolved.draftDepth) payload.spec_draft_n_max = resolved.draftDepth;
	if (load.nParallel) payload.n_parallel = load.nParallel;
	if (load.nBatch) payload.n_batch = load.nBatch;
	if (load.nUBatch) payload.n_ubatch = load.nUBatch;
	if (resolved.extraArgs.length > 0)
		payload.llama_extra_args = resolved.extraArgs;
	return payload;
}

/** Body for Unsloth's saved per-model load override. The model id stays qualified by its quant. */
export function buildModelOverridePayload(
	modelId: string,
	s: UnslothModelSettings,
): Record<string, unknown> {
	const load = s.load ?? {};
	const resolved = resolvedLoadSettings(load);
	const payload: Record<string, unknown> = {
		model_id: modelId,
		remove: false,
		llama_extra_args: resolved.extraArgs,
	};
	if (s.chatTemplate?.content)
		payload.chat_template_override = s.chatTemplate.content;
	if (load.maxSeqLength) payload.max_seq_length = load.maxSeqLength;
	if (load.cacheTypeKv) payload.kv_cache_dtype = load.cacheTypeKv;
	if (resolved.speculativeType)
		payload.speculative_type = resolved.speculativeType;
	if (resolved.draftDepth) payload.spec_draft_n_max = resolved.draftDepth;
	if (load.nParallel) payload.n_parallel = load.nParallel;
	if (load.nBatch) payload.n_batch = load.nBatch;
	if (load.nUBatch) payload.n_ubatch = load.nUBatch;
	return payload;
}

/** models.json samplingParams (OpenAI wire names), undefined fields omitted. */
export function buildSamplingParams(
	s?: SamplingSettings,
): Record<string, unknown> | undefined {
	if (!s) return undefined;
	const out: Record<string, unknown> = {};
	if (s.temperature !== undefined) out.temperature = s.temperature;
	if (s.topP !== undefined) out.top_p = s.topP;
	if (s.topK !== undefined) out.top_k = s.topK;
	if (s.minP !== undefined) out.min_p = s.minP;
	if (s.presencePenalty !== undefined) out.presence_penalty = s.presencePenalty;
	if (s.frequencyPenalty !== undefined)
		out.frequency_penalty = s.frequencyPenalty;
	if (s.repeatPenalty !== undefined) out.repetition_penalty = s.repeatPenalty;
	if (s.seed !== undefined) out.seed = s.seed;
	return Object.keys(out).length > 0 ? out : undefined;
}

/** models.json model entry for a configured Unsloth model. */
export function buildModelEntry(
	modelId: string,
	s: UnslothModelSettings,
	discovered?: { contextWindow?: number; maxTokens?: number },
): Record<string, unknown> {
	return {
		id: modelId,
		name: modelId,
		reasoning: s.thinking?.reasoning ?? false,
		input: ["text"],
		contextWindow: s.contextWindow ?? discovered?.contextWindow ?? 128000,
		maxTokens: s.maxTokens ?? discovered?.maxTokens ?? 16384,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		...(s.thinking?.compat ? { compat: s.thinking.compat } : {}),
		...(s.thinking?.thinkingLevelMap
			? { thinkingLevelMap: s.thinking.thinkingLevelMap }
			: {}),
		...(buildSamplingParams(s.sampling)
			? { samplingParams: buildSamplingParams(s.sampling) }
			: {}),
	};
}
