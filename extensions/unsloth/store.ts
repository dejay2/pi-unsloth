/**
 * Persistence helpers for ~/.pi/agent/models.json
 *
 * These are deliberately free of pi imports so they can be unit-tested
 * with plain node (type-stripping) — see test-store.ts.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

export const MODELS_JSON_PATH = join(homedir(), ".pi", "agent", "models.json");

export interface ModelEntry {
	id: string;
	name?: string;
	api?: string;
	reasoning?: boolean;
	input?: string[];
	contextWindow?: number;
	maxTokens?: number;
	cost?: { input: number; output: number; cacheRead: number; cacheWrite: number };
	compat?: Record<string, unknown>;
	[key: string]: unknown;
}

export interface ProviderEntry {
	baseUrl?: string;
	api?: string;
	apiKey?: string;
	headers?: Record<string, string>;
	authHeader?: boolean;
	compat?: Record<string, unknown>;
	models?: ModelEntry[];
	[key: string]: unknown;
}

export interface ModelsJson {
	providers: Record<string, ProviderEntry>;
	[key: string]: unknown;
}

/** Read models.json. Returns an empty skeleton when the file does not exist. Throws on corrupt JSON (never clobbers). */
export function readModelsJson(path: string = MODELS_JSON_PATH): ModelsJson {
	if (!existsSync(path)) return { providers: {} };
	const raw = readFileSync(path, "utf8").trim();
	if (!raw) return { providers: {} };
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch (err) {
		throw new Error(
			`Could not parse ${path}: ${err instanceof Error ? err.message : String(err)}. Fix or remove the file manually — refusing to overwrite it.`,
		);
	}
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error(`Unexpected top-level shape in ${path} (expected an object). Refusing to overwrite it.`);
	}
	const data = parsed as ModelsJson;
	if (data.providers === undefined || data.providers === null) {
		data.providers = {};
	} else if (typeof data.providers !== "object" || Array.isArray(data.providers)) {
		throw new Error(`Unexpected "providers" shape in ${path} (expected an object). Refusing to overwrite it.`);
	}
	return data;
}

/** Atomically write models.json (write to temp file, then rename). */
export function writeModelsJson(data: ModelsJson, path: string = MODELS_JSON_PATH): void {
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.tmp-${process.pid}`;
	writeFileSync(tmp, JSON.stringify(data, null, 2) + "\n", "utf8");
	renameSync(tmp, path);
}

/** Provider-level keys we merge when upserting. */
const PROVIDER_MERGE_KEYS = ["baseUrl", "api", "apiKey", "headers", "authHeader", "compat", "oauth"] as const;

/**
 * Insert or update a provider and its models.
 * - Provider-level fields from `cfg` overwrite existing ones (when defined).
 * - Models are upserted by `id`; existing models keep fields not present in the new entry.
 */
export function upsertProvider(data: ModelsJson, providerId: string, cfg: ProviderEntry): ModelsJson {
	const existing: ProviderEntry = data.providers[providerId] ?? {};
	const merged: ProviderEntry = { ...existing };
	for (const key of PROVIDER_MERGE_KEYS) {
		if (cfg[key] !== undefined) merged[key] = cfg[key];
	}
	const models: ModelEntry[] = Array.isArray(existing.models) ? [...existing.models] : [];
	for (const m of cfg.models ?? []) {
		const i = models.findIndex((x) => x.id === m.id);
		if (i >= 0) models[i] = { ...models[i], ...m };
		else models.push(m);
	}
	merged.models = models;
	data.providers[providerId] = merged;
	return data;
}

export interface RemoveResult {
	/** Model ids that were actually removed. */
	removed: string[];
	/** True when the provider entry was deleted entirely (no models left). */
	providerDeleted: boolean;
}

/**
 * Remove models from a provider. When the provider's model list becomes empty,
 * the whole provider entry is deleted (a models.json provider without models
 * is only meaningful as a built-in override, which this command does not manage).
 */
export function removeModels(data: ModelsJson, providerId: string, modelIds: string[]): RemoveResult {
	const provider = data.providers[providerId];
	if (!provider) return { removed: [], providerDeleted: false };
	const before = Array.isArray(provider.models) ? provider.models : [];
	const toRemove = new Set(modelIds);
	const kept = before.filter((m) => !toRemove.has(m.id));
	const removed = before.filter((m) => toRemove.has(m.id)).map((m) => m.id);
	if (kept.length === 0) {
		delete data.providers[providerId];
		return { removed, providerDeleted: true };
	}
	provider.models = kept;
	return { removed, providerDeleted: false };
}

/** Remove an entire provider entry. Returns false when it did not exist. */
export function removeProvider(data: ModelsJson, providerId: string): boolean {
	if (!(providerId in data.providers)) return false;
	delete data.providers[providerId];
	return true;
}
