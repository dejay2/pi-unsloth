/**
 * Model discovery: fetch the list of models a custom endpoint serves.
 *
 * Supports the /models listing conventions of OpenAI-compatible servers
 * (Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, …), Anthropic-compatible
 * endpoints, and Google Generative AI. No pi imports — unit-testable.
 */

export interface DiscoveredModel {
	id: string;
	/** Present when the server advertises it (some OpenAI-compatible servers do). */
	contextWindow?: number;
	maxTokens?: number;
	/** Unsloth Studio advertises a single preferred on-disk quant per repo. */
	quant?: string;
	/** Unsloth Studio marks the currently-resident model(s). */
	loaded?: boolean;
	displayName?: string;
}

export interface DiscoveryConfig {
	baseUrl: string;
	api: string;
	/** Resolved literal key, or undefined for keyless/unresolvable. */
	apiKey?: string;
}

/** Resolve a models.json apiKey value to a literal key. Never executes `!command` values. */
export function resolveApiKey(apiKey: string | undefined, env: Record<string, string | undefined> = process.env): string | undefined {
	if (!apiKey || apiKey === "keyless") return undefined;
	if (apiKey.startsWith("!")) return undefined; // shell command — do not execute
	if (apiKey.startsWith("$")) {
		// $VAR or ${VAR} — only a bare reference is resolvable here
		const m = /^\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?$/.exec(apiKey);
		return m ? env[m[1]] : undefined;
	}
	return apiKey;
}

/** Candidate URLs for the models listing, in order. */
export function modelListUrls(cfg: DiscoveryConfig): string[] {
	const base = cfg.baseUrl.replace(/\/+$/, "");
	if (cfg.api === "google-generative-ai") {
		const key = cfg.apiKey ? `?key=${encodeURIComponent(cfg.apiKey)}` : "";
		return [`${base}/models${key}`];
	}
	if (cfg.api === "anthropic-messages") {
		// Anthropic serves GET /v1/models; proxies differ on whether baseUrl includes /v1
		return base.endsWith("/v1") ? [`${base}/models`] : [`${base}/v1/models`, `${base}/models`];
	}
	// openai-completions / openai-responses and anything else OpenAI-shaped
	return [`${base}/models`];
}

function headersFor(cfg: DiscoveryConfig): Record<string, string> {
	const headers: Record<string, string> = { accept: "application/json" };
	if (cfg.api === "anthropic-messages") {
		if (cfg.apiKey) headers["x-api-key"] = cfg.apiKey;
		headers["anthropic-version"] = "2023-06-01";
	} else if (cfg.api !== "google-generative-ai" && cfg.apiKey) {
		headers.authorization = `Bearer ${cfg.apiKey}`;
	}
	return headers;
}

/** Parse a /models response body into model entries. Throws on unrecognized shape. */
export function parseModelsResponse(api: string, body: unknown): DiscoveredModel[] {
	if (body === null || typeof body !== "object") throw new Error("Unexpected response shape");
	const obj = body as Record<string, unknown>;

	if (api === "google-generative-ai") {
		if (!Array.isArray(obj.models)) throw new Error("Unexpected Google /models shape");
		return obj.models
			.map((m) => {
				const name = String((m as Record<string, unknown>).name ?? "");
				const entry: DiscoveredModel = { id: name.replace(/^models\//, "") };
				const inputLimit = (m as Record<string, unknown>).inputTokenLimit;
				const outputLimit = (m as Record<string, unknown>).outputTokenLimit;
				if (typeof inputLimit === "number") entry.contextWindow = inputLimit;
				if (typeof outputLimit === "number") entry.maxTokens = outputLimit;
				return entry;
			})
			.filter((m) => m.id.length > 0);
	}

	// OpenAI and Anthropic both answer { data: [{ id, ... }] }
	if (!Array.isArray(obj.data)) throw new Error("Unexpected /models shape (missing data array)");
	return obj.data
		.map((m) => {
			const rec = m as Record<string, unknown>;
			const entry: DiscoveredModel = { id: String(rec.id ?? "") };
			// Some OpenAI-compatible servers (LM Studio, vLLM plugins) advertise limits
			const cw = rec.context_window ?? rec.context_length ?? rec.max_context_length;
			const mt = rec.max_tokens ?? rec.max_output_tokens;
			if (typeof cw === "number" && cw > 0) entry.contextWindow = cw;
			if (typeof mt === "number" && mt > 0) entry.maxTokens = mt;
			// Unsloth Studio extras
			if (typeof rec.quant === "string" && rec.quant) entry.quant = rec.quant;
			if (typeof rec.loaded === "boolean") entry.loaded = rec.loaded;
			if (typeof rec.display_name === "string" && rec.display_name) entry.displayName = rec.display_name;
			return entry;
		})
		.filter((m) => m.id.length > 0);
}

/**
 * Fetch the model list from an endpoint. Returns null when discovery is not
 * possible (network error, non-JSON, unrecognized shape) — callers fall back
 * to manual entry.
 */
export async function fetchModels(cfg: DiscoveryConfig, timeoutMs = 10_000): Promise<DiscoveredModel[] | null> {
	const urls = modelListUrls(cfg);
	const headers = headersFor(cfg);
	for (const url of urls) {
		try {
			const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
			if (!res.ok) continue;
			const body = await res.json();
			const models = parseModelsResponse(cfg.api, body);
			if (models.length > 0) return models;
		} catch {
			// try next candidate URL, then give up
		}
	}
	return null;
}

// ---------------------------------------------------------------------------
// Unsloth Studio quant expansion
//
// Unsloth's OpenAI-compatible GET /v1/models advertises ONE entry per model
// repo (keyed by clean id) with a single `quant` hint — even when several
// quants of that repo are downloaded. Its clients pin a quant by requesting
// "<id>:<quant>". The Studio API exposes every downloaded quant at
// GET /api/models/gguf-variants?repo_id=<org/repo>, so we can expand the
// catalog into one selectable entry per downloaded quant.
// ---------------------------------------------------------------------------

export interface GgufVariant {
	quant: string;
	sizeBytes?: number;
}

const HF_REPO_RE = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

/** Strip a trailing /v1 (or /v1beta) to reach the server root. */
export function serverRoot(baseUrl: string): string {
	return baseUrl.replace(/\/+$/, "").replace(/\/v1(beta)?$/i, "");
}

/**
 * Fetch downloaded GGUF variants for a repo from an Unsloth Studio server.
 * Returns null when the server is not Unsloth / the route is unavailable.
 */
export async function fetchGgufVariants(
	cfg: DiscoveryConfig,
	repoId: string,
	timeoutMs = 10_000,
): Promise<GgufVariant[] | null> {
	const url = `${serverRoot(cfg.baseUrl)}/api/models/gguf-variants?repo_id=${encodeURIComponent(repoId)}`;
	try {
		const res = await fetch(url, { headers: headersFor(cfg), signal: AbortSignal.timeout(timeoutMs) });
		if (!res.ok) return null;
		const body = (await res.json()) as Record<string, unknown>;
		if (!Array.isArray(body.variants)) return null;
		return body.variants
			.map((v) => {
				const rec = v as Record<string, unknown>;
				return {
					quant: String(rec.quant ?? ""),
					sizeBytes: typeof rec.size_bytes === "number" ? rec.size_bytes : undefined,
					downloaded: Boolean(rec.downloaded),
				};
			})
			.filter((v) => v.downloaded && v.quant)
			.map(({ quant, sizeBytes }) => ({ quant, sizeBytes }));
	} catch {
		return null;
	}
}

function formatGb(bytes?: number): string | undefined {
	if (!bytes || bytes <= 0) return undefined;
	return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * Expand an Unsloth catalog into per-quant entries. Only runs when the
 * catalog carries Unsloth's `quant` signature; each HF-repo-shaped id is
 * probed for its downloaded variants and replaced by `<id>:<quant>` entries.
 * Servers that don't answer the probe keep their original entries.
 */
export async function expandUnslothQuants(cfg: DiscoveryConfig, models: DiscoveredModel[]): Promise<DiscoveredModel[]> {
	if (!models.some((m) => m.quant !== undefined)) return models;
	const out: DiscoveredModel[] = [];
	for (const m of models) {
		if (!HF_REPO_RE.test(m.id)) {
			out.push(m);
			continue;
		}
		const variants = await fetchGgufVariants(cfg, m.id);
		if (!variants || variants.length === 0) {
			out.push(m);
			continue;
		}
		for (const v of variants) {
			const size = formatGb(v.sizeBytes);
			// A loaded repo only has ONE resident quant: Unsloth sets the catalog
			// entry's `quant` to the resident one — only that variant is loaded.
			const isLoaded = m.loaded === true && m.quant === v.quant;
			out.push({
				...m,
				id: `${m.id}:${v.quant}`,
				quant: v.quant,
				loaded: isLoaded,
				displayName: `${m.displayName ?? m.id}:${v.quant}${size ? ` (${size})` : ""}${isLoaded ? " (loaded)" : ""}`,
			});
		}
	}
	return out;
}
