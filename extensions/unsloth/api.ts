/**
 * Unsloth server API client. No pi imports — unit-testable with a mock server.
 */

import {
	fetchModels,
	expandUnslothQuants,
	serverRoot,
	type DiscoveredModel,
	type DiscoveryConfig,
} from "./discover.ts";
import { fetchUnslothReasoning, type ReasoningInfo } from "./thinking.ts";
import {
	buildModelOverridePayload,
	splitModelRef,
	type UnslothModelSettings,
} from "./config.ts";

export interface UnslothStatus {
	activeModel?: string;
	modelIdentifier?: string;
	ggufVariant?: string;
	loaded: string[];
	chatTemplate?: string;
	chatTemplateOverride?: string;
	/** Studio retried without the requested draft helper after a server startup failure. */
	specFallbackReason?: string;
	reasoning: ReasoningInfo | null;
	raw: Record<string, unknown>;
}

export async function fetchStatus(
	cfg: DiscoveryConfig,
	timeoutMs = 8_000,
): Promise<UnslothStatus | null> {
	const url = `${serverRoot(cfg.baseUrl)}/api/inference/status`;
	try {
		const headers: Record<string, string> = { accept: "application/json" };
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, {
			headers,
			signal: AbortSignal.timeout(timeoutMs),
		});
		if (!res.ok) return null;
		const raw = (await res.json()) as Record<string, unknown>;
		if (typeof raw.supports_reasoning !== "boolean" && !raw.active_model)
			return null;
		const reasoning = await fetchUnslothReasoning(cfg, timeoutMs);
		return {
			activeModel:
				typeof raw.active_model === "string" ? raw.active_model : undefined,
			modelIdentifier:
				typeof raw.model_identifier === "string" ? raw.model_identifier : undefined,
			ggufVariant:
				typeof raw.gguf_variant === "string" ? raw.gguf_variant : undefined,
			loaded: Array.isArray(raw.loaded)
				? raw.loaded.filter((x): x is string => typeof x === "string")
				: [],
			...(typeof raw.chat_template === "string"
				? { chatTemplate: raw.chat_template }
				: {}),
			...(typeof raw.chat_template_override === "string"
				? { chatTemplateOverride: raw.chat_template_override }
				: {}),
			...(typeof raw.spec_fallback_reason === "string"
				? { specFallbackReason: raw.spec_fallback_reason }
				: {}),
			reasoning,
			raw,
		};
	} catch {
		return null;
	}
}

/** List all models with per-quant expansion (Unsloth-aware). */
export async function listModels(
	cfg: DiscoveryConfig,
): Promise<DiscoveredModel[] | null> {
	const models = await fetchModels(cfg);
	if (!models) return null;
	return expandUnslothQuants(cfg, models);
}

export interface LoadResult {
	ok: boolean;
	status: number;
	error?: string;
}

/** Save structural settings in Unsloth's per-model override store. */
export async function saveModelOverride(
	cfg: DiscoveryConfig,
	modelId: string,
	settings: UnslothModelSettings,
	timeoutMs = 10_000,
): Promise<LoadResult> {
	const url = `${serverRoot(cfg.baseUrl)}/api/settings/openai-auto-switch/overrides`;
	try {
		const headers: Record<string, string> = {
			"content-type": "application/json",
			accept: "application/json",
		};
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, {
			method: "PUT",
			headers,
			body: JSON.stringify(buildModelOverridePayload(modelId, settings)),
			signal: AbortSignal.timeout(timeoutMs),
		});
		if (res.ok) return { ok: true, status: res.status };
		let detail = "";
		try {
			const body = await res.json();
			detail = body?.detail ?? body?.error ?? JSON.stringify(body);
		} catch {
			detail = await res.text().catch(() => "");
		}
		return { ok: false, status: res.status, error: String(detail).slice(0, 300) };
	} catch (err) {
		return {
			ok: false,
			status: 0,
			error: err instanceof Error ? err.message : String(err),
		};
	}
}

export type AutoSwitchSettings = Record<string, unknown> & { enabled: boolean };

export async function saveLastLocalModel(
	cfg: DiscoveryConfig,
	modelId: string,
	clientNow = Date.now(),
	timeoutMs = 10_000,
): Promise<LoadResult> {
	const { modelPath, variant } = splitModelRef(modelId);
	const url = `${serverRoot(cfg.baseUrl)}/api/settings/last-local-model`;
	try {
		const headers: Record<string, string> = {
			"content-type": "application/json",
			accept: "application/json",
		};
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, {
			method: "PUT",
			headers,
			body: JSON.stringify({
				id: modelPath,
				kind: "gguf",
				...(variant ? { gguf_variant: variant } : {}),
				loaded_at: clientNow,
				client_now: clientNow,
			}),
			signal: AbortSignal.timeout(timeoutMs),
		});
		return res.ok
			? { ok: true, status: res.status }
			: { ok: false, status: res.status, error: (await res.text()).slice(0, 300) };
	} catch (err) {
		return {
			ok: false,
			status: 0,
			error: err instanceof Error ? err.message : String(err),
		};
	}
}

export async function fetchAutoSwitch(
	cfg: DiscoveryConfig,
	timeoutMs = 10_000,
): Promise<AutoSwitchSettings | null> {
	const url = `${serverRoot(cfg.baseUrl)}/api/settings/openai-auto-switch`;
	try {
		const headers: Record<string, string> = { accept: "application/json" };
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, {
			headers,
			signal: AbortSignal.timeout(timeoutMs),
		});
		if (!res.ok) return null;
		const body: unknown = await res.json();
		if (
			!body ||
			typeof body !== "object" ||
			Array.isArray(body) ||
			!("enabled" in body) ||
			typeof body.enabled !== "boolean"
		) {
			return null;
		}
		return { ...body, enabled: body.enabled };
	} catch {
		return null;
	}
}

export async function setAutoSwitch(
	cfg: DiscoveryConfig,
	enabled: boolean,
	timeoutMs = 10_000,
): Promise<LoadResult> {
	const current = await fetchAutoSwitch(cfg, timeoutMs);
	if (!current)
		return {
			ok: false,
			status: 0,
			error: "Could not read the current automatic model switching settings",
		};
	const url = `${serverRoot(cfg.baseUrl)}/api/settings/openai-auto-switch`;
	try {
		const headers: Record<string, string> = {
			"content-type": "application/json",
			accept: "application/json",
		};
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, {
			method: "PUT",
			headers,
			body: JSON.stringify({ ...current, enabled }),
			signal: AbortSignal.timeout(timeoutMs),
		});
		return res.ok
			? { ok: true, status: res.status }
			: { ok: false, status: res.status, error: (await res.text()).slice(0, 300) };
	} catch (err) {
		return {
			ok: false,
			status: 0,
			error: err instanceof Error ? err.message : String(err),
		};
	}
}

/** POST /api/inference/load — (re)load a model with structural settings. */
export async function loadModel(
	cfg: DiscoveryConfig,
	payload: Record<string, unknown>,
	timeoutMs = 600_000, // loads can take minutes
): Promise<LoadResult> {
	const url = `${serverRoot(cfg.baseUrl)}/api/inference/load`;
	try {
		const headers: Record<string, string> = {
			"content-type": "application/json",
			accept: "application/json",
		};
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, {
			method: "POST",
			headers,
			body: JSON.stringify(payload),
			signal: AbortSignal.timeout(timeoutMs),
		});
		if (res.ok) return { ok: true, status: res.status };
		let detail = "";
		try {
			const body = await res.json();
			detail = body?.detail ?? body?.error ?? JSON.stringify(body);
		} catch {
			detail = await res.text().catch(() => "");
		}
		return { ok: false, status: res.status, error: String(detail).slice(0, 300) };
	} catch (err) {
		return {
			ok: false,
			status: 0,
			error: err instanceof Error ? err.message : String(err),
		};
	}
}
