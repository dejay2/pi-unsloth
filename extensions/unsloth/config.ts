/**
 * Per-provider/model settings store for Unsloth servers, plus the pure
 * payload builders that turn settings into Unsloth/pi wire formats.
 *
 * Lives at ~/.pi/agent/unsloth.json. No pi imports — unit-testable.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
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

/** Load-time (structural) llama.cpp settings, applied via POST /api/inference/load. */
export interface LoadSettings {
	maxSeqLength?: number;
	cacheTypeKv?: string;
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
	models: Record<string, UnslothModelSettings>;
}

export interface UnslothConfig {
	providers: Record<string, UnslothProviderConfig>;
}

export function readConfig(path: string = CONFIG_PATH): UnslothConfig {
	if (!existsSync(path)) return { providers: {} };
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8"));
		if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { providers: {} };
		if (!parsed.providers || typeof parsed.providers !== "object") parsed.providers = {};
		return parsed as UnslothConfig;
	} catch {
		return { providers: {} };
	}
}

export function writeConfig(data: UnslothConfig, path: string = CONFIG_PATH): void {
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
	const p = (data.providers[providerId] ??= { baseUrl: provider.baseUrl, apiKey: provider.apiKey, models: {} });
	p.baseUrl = provider.baseUrl;
	p.apiKey = provider.apiKey;
	p.models[modelId] = settings;
}

export function removeModel(data: UnslothConfig, providerId: string, modelId: string): boolean {
	const p = data.providers[providerId];
	if (!p || !(modelId in p.models)) return false;
	delete p.models[modelId];
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
export function splitModelRef(modelId: string): { modelPath: string; variant?: string } {
	const i = modelId.lastIndexOf(":");
	if (i <= 0) return { modelPath: modelId };
	return { modelPath: modelId.slice(0, i), variant: modelId.slice(i + 1) };
}

/** Body for POST /api/inference/load (== /v1/load). Undefined/empty fields omitted. */
export function buildLoadPayload(modelId: string, s: UnslothModelSettings): Record<string, unknown> {
	const { modelPath, variant } = splitModelRef(modelId);
	const load = s.load ?? {};
	const payload: Record<string, unknown> = { model_path: modelPath };
	if (variant) payload.gguf_variant = variant;
	if (s.chatTemplate?.content) payload.chat_template_override = s.chatTemplate.content;
	if (load.maxSeqLength) payload.max_seq_length = load.maxSeqLength;
	if (load.cacheTypeKv) payload.cache_type_kv = load.cacheTypeKv;
	if (load.speculativeType) payload.speculative_type = load.speculativeType;
	if (load.specDraftNMax) payload.spec_draft_n_max = load.specDraftNMax;
	if (load.nParallel) payload.n_parallel = load.nParallel;
	if (load.nBatch) payload.n_batch = load.nBatch;
	if (load.nUBatch) payload.n_ubatch = load.nUBatch;
	if (load.extraArgs && load.extraArgs.length > 0) payload.llama_extra_args = load.extraArgs;
	return payload;
}

/** models.json samplingParams (OpenAI wire names), undefined fields omitted. */
export function buildSamplingParams(s?: SamplingSettings): Record<string, unknown> | undefined {
	if (!s) return undefined;
	const out: Record<string, unknown> = {};
	if (s.temperature !== undefined) out.temperature = s.temperature;
	if (s.topP !== undefined) out.top_p = s.topP;
	if (s.topK !== undefined) out.top_k = s.topK;
	if (s.minP !== undefined) out.min_p = s.minP;
	if (s.presencePenalty !== undefined) out.presence_penalty = s.presencePenalty;
	if (s.frequencyPenalty !== undefined) out.frequency_penalty = s.frequencyPenalty;
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
		...(s.thinking?.thinkingLevelMap ? { thinkingLevelMap: s.thinking.thinkingLevelMap } : {}),
		...(buildSamplingParams(s.sampling) ? { samplingParams: buildSamplingParams(s.sampling) } : {}),
	};
}
