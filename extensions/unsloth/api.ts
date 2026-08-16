/**
 * Unsloth server API client. No pi imports — unit-testable with a mock server.
 */

import { fetchModels, expandUnslothQuants, serverRoot, type DiscoveredModel, type DiscoveryConfig } from "./discover.ts";
import { fetchUnslothReasoning, type ReasoningInfo } from "./thinking.ts";

export interface UnslothStatus {
	activeModel?: string;
	loaded: string[];
	reasoning: ReasoningInfo | null;
	raw: Record<string, unknown>;
}

export async function fetchStatus(cfg: DiscoveryConfig, timeoutMs = 8_000): Promise<UnslothStatus | null> {
	const url = `${serverRoot(cfg.baseUrl)}/api/inference/status`;
	try {
		const headers: Record<string, string> = { accept: "application/json" };
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
		if (!res.ok) return null;
		const raw = (await res.json()) as Record<string, unknown>;
		if (typeof raw.supports_reasoning !== "boolean" && !raw.active_model) return null;
		const reasoning = await fetchUnslothReasoning(cfg, timeoutMs);
		return {
			activeModel: typeof raw.active_model === "string" ? raw.active_model : undefined,
			loaded: Array.isArray(raw.loaded) ? raw.loaded.filter((x): x is string => typeof x === "string") : [],
			reasoning,
			raw,
		};
	} catch {
		return null;
	}
}

/** List all models with per-quant expansion (Unsloth-aware). */
export async function listModels(cfg: DiscoveryConfig): Promise<DiscoveredModel[] | null> {
	const models = await fetchModels(cfg);
	if (!models) return null;
	return expandUnslothQuants(cfg, models);
}

export interface LoadResult {
	ok: boolean;
	status: number;
	error?: string;
}

/** POST /api/inference/load — (re)load a model with structural settings. */
export async function loadModel(
	cfg: DiscoveryConfig,
	payload: Record<string, unknown>,
	timeoutMs = 600_000, // loads can take minutes
): Promise<LoadResult> {
	const url = `${serverRoot(cfg.baseUrl)}/api/inference/load`;
	try {
		const headers: Record<string, string> = { "content-type": "application/json", accept: "application/json" };
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
		return { ok: false, status: 0, error: err instanceof Error ? err.message : String(err) };
	}
}
