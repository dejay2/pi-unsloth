/**
 * pi-unsloth — first-class Unsloth Studio integration for pi.
 *
 * /login: "Unsloth (local server)" appears in pi's native login menu (both
 * auth categories). The wizard asks for the server URL + API key, discovers
 * models (all downloaded quants), lets you pick one, then configures:
 *   - thinking control (auto-detected from Unsloth's template classification)
 *   - load-time llama.cpp settings (context, KV dtype, MTP/speculative,
 *     parallel slots, extra args) — applied via POST /api/inference/load
 *     whenever you switch to the model in pi
 *   - sampling (temperature/top_p/top_k/min_p/repeat penalty/seed) — written
 *     to models.json samplingParams
 *   - thinking-conditional sampling — swapped in per request via a
 *     before_provider_request hook when pi's thinking level is not "off"
 *
 * /unsloth: manage everything afterwards — add models (multi-select),
 * reconfigure, apply settings now, server status, remove.
 */

import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { isKeyRelease, matchesKey, type KeyId } from "@earendil-works/pi-tui";
import * as piAi from "@earendil-works/pi-ai";
import type { ProviderStreams } from "@earendil-works/pi-ai";

import {
	readConfig,
	writeConfig,
	upsertModel,
	removeModel as removeModelFromConfig,
	buildLoadPayload,
	buildModelEntry,
	buildSamplingParams,
	splitModelRef,
	CONFIG_PATH,
	type UnslothModelSettings,
	type SamplingSettings,
} from "./config.ts";
import { fetchStatus, listModels, loadModel } from "./api.ts";
import { resolveApiKey, type DiscoveredModel } from "./discover.ts";
import { fetchUnslothReasoning, type ReasoningInfo } from "./thinking.ts";
import {
	providerSetup,
	pickModel,
	settingsWizard,
	adaptAuthInteraction,
	type WizardInteraction,
} from "./wizard.ts";
import { MultiSelect } from "./multiselect.ts";
import { readModelsJson, writeModelsJson, upsertProvider, type ModelsJson } from "./store.ts";

const PROVIDER_ID = "unsloth";
const VEHICLE_ID = "unsloth-server";

let sessionCtx: ExtensionContext | undefined;

function discoveryCfg(baseUrl: string, apiKey: string) {
	return { baseUrl, api: "openai-completions", apiKey: resolveApiKey(apiKey) };
}

/** Make the provider's configured models live in pi immediately. */
function registerProviderFromFile(pi: ExtensionAPI, providerId: string, data: ModelsJson): void {
	const p = data.providers[providerId];
	if (!p) return;
	pi.registerProvider(providerId, {
		...(p.baseUrl ? { baseUrl: p.baseUrl } : {}),
		...(p.api ? { api: p.api } : {}),
		...(p.apiKey ? { apiKey: p.apiKey } : {}),
		models: (Array.isArray(p.models) ? p.models : []).map((m) => ({
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 16384,
			...m,
			name: m.name ?? m.id,
		})),
	});
}

/** Persist a configured model to both unsloth.json and models.json, then go live. */
function saveModel(
	pi: ExtensionAPI,
	provider: { baseUrl: string; apiKey: string },
	model: DiscoveredModel,
	settings: UnslothModelSettings,
	configPath?: string,
): void {
	const cfg = readConfig(configPath);
	upsertModel(cfg, PROVIDER_ID, provider, model.id, settings);
	writeConfig(cfg, configPath);

	const data = readModelsJson();
	upsertProvider(data, PROVIDER_ID, {
		baseUrl: provider.baseUrl,
		api: "openai-completions",
		apiKey: provider.apiKey,
		compat: { supportsDeveloperRole: false },
		models: [buildModelEntry(model.id, settings, model)],
	});
	writeModelsJson(data);
	registerProviderFromFile(pi, PROVIDER_ID, data);
}

/** Apply load-time settings on the server (fire-and-forget with notifies). */
async function applyLoadSettings(
	ctx: ExtensionContext,
	provider: { baseUrl: string; apiKey: string },
	modelId: string,
	settings: UnslothModelSettings,
): Promise<void> {
	const payload = buildLoadPayload(modelId, settings);
	ctx.ui.notify(`Unsloth: loading ${modelId} with configured settings…`, "info");
	const res = await loadModel(discoveryCfg(provider.baseUrl, provider.apiKey), payload);
	if (res.ok) {
		ctx.ui.notify(`Unsloth: ${modelId} loaded`, "info");
	} else if (res.status === 409) {
		ctx.ui.notify(`Unsloth: load refused (server busy, 409). ${res.error ?? ""}`, "warning");
	} else {
		ctx.ui.notify(`Unsloth: load failed — ${res.error ?? `HTTP ${res.status}`}`, "error");
	}
}

// ---------------------------------------------------------------------------
// /login flow
// ---------------------------------------------------------------------------

async function loginFlow(pi: ExtensionAPI, ui: WizardInteraction): Promise<string> {
	const provider = await providerSetup(ui);
	const cfg = discoveryCfg(provider.baseUrl, provider.apiKey);

	ui.progress("Fetching models…");
	const models = await listModels(cfg);
	if (!models || models.length === 0) {
		throw new Error("Could not reach the server or it has no models. Check URL and key.");
	}

	const model = await pickModel(ui, models);
	ui.progress("Detecting thinking control…");
	const detected = await fetchUnslothReasoning(cfg);
	const settings = await settingsWizard(ui, model, detected);

	saveModel(pi, provider, model, settings);

	// Apply load-time settings immediately so the server matches the config.
	const payload = buildLoadPayload(model.id, settings);
	ui.progress("Applying load-time settings on the server…");
	const res = await loadModel(cfg, payload);
	if (!res.ok) ui.progress(`Note: load returned ${res.error ?? `HTTP ${res.status}`} — settings apply on next switch.`);

	// Best-effort: switch to the new model.
	try {
		const m = sessionCtx?.modelRegistry.find(PROVIDER_ID, model.id);
		if (m) await pi.setModel(m);
	} catch {
		// non-fatal
	}

	return resolveApiKey(provider.apiKey) ?? "keyless";
}

function registerLoginVehicle(pi: ExtensionAPI): void {
	// NOTE: named pi-ai value imports are unreliable through pi's loader (jiti
	// interop) — use the namespace object.
	const createProvider = (piAi as any).createProvider as (input: any) => any;
	const noStreams = {
		stream() {
			throw new Error("unsloth vehicle has no models");
		},
		streamSimple() {
			throw new Error("unsloth vehicle has no models");
		},
	} as unknown as ProviderStreams;

	const vehicle = createProvider({
		id: VEHICLE_ID,
		name: " Unsloth (local server)",
		baseUrl: "http://127.0.0.1:9",
		models: [],
		api: noStreams,
		auth: {
			apiKey: {
				name: "Unsloth server setup",
				async login(interaction: any) {
					const key = await loginFlow(pi, adaptAuthInteraction(interaction));
					return { type: "api_key" as const, key };
				},
				async resolve({ credential }: any) {
					return credential?.key
						? { auth: { apiKey: credential.key }, source: "pi-unsloth login" }
						: undefined;
				},
			},
			oauth: {
				name: "Unsloth server setup",
				async login(interaction: any) {
					const key = await loginFlow(pi, adaptAuthInteraction(interaction));
					return { type: "oauth" as const, access: key, refresh: key, expires: Number.MAX_SAFE_INTEGER };
				},
				async refresh(credential: any) {
					return credential;
				},
				async toAuth(credential: any) {
					return { apiKey: credential.access };
				},
			},
		},
	});
	pi.registerProvider(vehicle);
}

// ---------------------------------------------------------------------------
// /unsloth command
// ---------------------------------------------------------------------------

function adaptUi(ctx: ExtensionCommandContext): WizardInteraction {
	return {
		text: async (message, placeholder) => {
			const r = await ctx.ui.input(message, placeholder);
			if (r === undefined) throw new Error("Cancelled");
			return r;
		},
		select: async (message, options) => {
			const r = await ctx.ui.select(message, options.map((o) => o.label));
			if (r === undefined) throw new Error("Cancelled");
			return options.find((o) => o.label === r)?.id;
		},
		progress: (message) => ctx.ui.notify(message, "info"),
	};
}

function getProvider(ctx: ExtensionCommandContext): { baseUrl: string; apiKey: string } | undefined {
	const p = readConfig().providers[PROVIDER_ID];
	if (p) return p;
	// Fall back to models.json (provider exists but no unsloth.json yet)
	try {
		const mj = readModelsJson().providers[PROVIDER_ID];
		if (mj?.baseUrl && mj?.apiKey) return { baseUrl: mj.baseUrl, apiKey: mj.apiKey };
	} catch {
		// ignore
	}
	return undefined;
}

async function cmdAddModels(pi: ExtensionAPI, ctx: ExtensionCommandContext): Promise<void> {
	let provider = getProvider(ctx);
	if (!provider) {
		ctx.ui.notify("No Unsloth server configured yet — set one up:", "info");
		const ui = adaptUi(ctx);
		provider = await providerSetup(ui);
	}
	const cfg = discoveryCfg(provider.baseUrl, provider.apiKey);
	ctx.ui.notify("Fetching models…", "info");
	const models = await listModels(cfg);
	if (!models || models.length === 0) {
		ctx.ui.notify("Could not fetch models — check server URL/key.", "error");
		return;
	}

	const labels = models.map((m) => m.displayName ?? m.id);
	const chosen = await ctx.ui.custom<string[] | null>((tui, theme, _kb, done) => {
		const ms = new MultiSelect(
			labels,
			Math.min(labels.length, 12),
			{
				accent: (s: string) => theme.fg("accent", s),
				muted: (s: string) => theme.fg("muted", s),
				dim: (s: string) => theme.fg("dim", s),
				bold: (s: string) => theme.bold(s),
				warning: (s: string) => theme.fg("warning", s),
			},
			done,
			(data, keyId) => matchesKey(data, keyId as KeyId),
		);
		return {
			render: (w: number) => ms.render(w),
			invalidate: () => ms.invalidate(),
			handleInput: (data: string) => {
				if (isKeyRelease(data)) return;
				ms.handleInput(data);
				tui.requestRender();
			},
		};
	});
	if (!chosen || chosen.length === 0) return;
	const picked = models.filter((m) => chosen.includes(m.displayName ?? m.id));

	ctx.ui.notify("Detecting thinking control…", "info");
	const detected = await fetchUnslothReasoning(cfg);
	const ui = adaptUi(ctx);
	for (const model of picked) {
		ctx.ui.notify(`Settings for ${model.id}`, "info");
		const settings = await settingsWizard(ui, model, detected);
		saveModel(pi, provider, model, settings);
	}
	ctx.ui.notify(`Added ${picked.length} model(s). Load-time settings apply when you switch to each model.`, "info");
	await offerSwitch(pi, ctx, picked[0].id);
}

async function cmdConfigure(pi: ExtensionAPI, ctx: ExtensionCommandContext): Promise<void> {
	const provider = getProvider(ctx);
	if (!provider) {
		ctx.ui.notify("No Unsloth server configured.", "error");
		return;
	}
	const cfg = readConfig();
	const modelIds = Object.keys(cfg.providers[PROVIDER_ID]?.models ?? {});
	if (modelIds.length === 0) {
		ctx.ui.notify("No configured models — use Add models first.", "warning");
		return;
	}
	const choice = await ctx.ui.select("Configure which model?", modelIds);
	if (!choice) return;
	const existing = cfg.providers[PROVIDER_ID].models[choice];

	ctx.ui.notify("Detecting thinking control…", "info");
	const detected = await fetchUnslothReasoning(discoveryCfg(provider.baseUrl, provider.apiKey));
	const model: DiscoveredModel = { id: choice, contextWindow: existing.contextWindow, maxTokens: existing.maxTokens };
	const settings = await settingsWizard(adaptUi(ctx), model, detected, existing);
	saveModel(pi, provider, model, settings);
	ctx.ui.notify(`Saved settings for ${choice}.`, "info");
}

async function cmdApplyNow(ctx: ExtensionCommandContext): Promise<void> {
	const provider = getProvider(ctx);
	if (!provider) {
		ctx.ui.notify("No Unsloth server configured.", "error");
		return;
	}
	const cfg = readConfig();
	const modelIds = Object.keys(cfg.providers[PROVIDER_ID]?.models ?? {});
	if (modelIds.length === 0) {
		ctx.ui.notify("No configured models.", "warning");
		return;
	}
	const choice = await ctx.ui.select("Apply settings + (re)load which model?", modelIds);
	if (!choice) return;
	await applyLoadSettings(ctx, provider, choice, cfg.providers[PROVIDER_ID].models[choice]);
}

async function cmdStatus(ctx: ExtensionCommandContext): Promise<void> {
	const provider = getProvider(ctx);
	if (!provider) {
		ctx.ui.notify("No Unsloth server configured.", "error");
		return;
	}
	const status = await fetchStatus(discoveryCfg(provider.baseUrl, provider.apiKey));
	if (!status) {
		ctx.ui.notify("Could not reach the server.", "error");
		return;
	}
	const lines = [
		`Server: ${provider.baseUrl}`,
		`Loaded: ${status.activeModel ?? "(none)"}`,
	];
	if (status.reasoning) {
		lines.push(
			`Thinking: ${status.reasoning.style}${status.reasoning.levels.length ? ` — levels: ${status.reasoning.levels.join(", ")}` : ""}`,
		);
	}
	const configured = Object.keys(readConfig().providers[PROVIDER_ID]?.models ?? {});
	lines.push(`Configured in pi: ${configured.length} model(s)`);
	ctx.ui.setWidget("unsloth-status", lines);
}

async function cmdRemove(pi: ExtensionAPI, ctx: ExtensionCommandContext): Promise<void> {
	const cfg = readConfig();
	const modelIds = Object.keys(cfg.providers[PROVIDER_ID]?.models ?? {});
	if (modelIds.length === 0) {
		ctx.ui.notify("No configured models.", "warning");
		return;
	}
	const choice = await ctx.ui.select("Remove which model from pi?", modelIds);
	if (!choice) return;
	const yes = await ctx.ui.confirm("Remove model?", `Remove ${choice} from pi? (Files on the server are untouched.)`);
	if (!yes) return;
	removeModelFromConfig(cfg, PROVIDER_ID, choice);
	writeConfig(cfg);
	const data = readModelsJson();
	const p = data.providers[PROVIDER_ID];
	if (p?.models) {
		p.models = p.models.filter((m) => m.id !== choice);
		if (p.models.length === 0) delete data.providers[PROVIDER_ID];
	}
	writeModelsJson(data);
	if (data.providers[PROVIDER_ID]) registerProviderFromFile(pi, PROVIDER_ID, data);
	else pi.unregisterProvider(PROVIDER_ID);
	ctx.ui.notify(`Removed ${choice}.`, "info");
}

async function offerSwitch(pi: ExtensionAPI, ctx: ExtensionCommandContext, modelId: string): Promise<void> {
	const yes = await ctx.ui.confirm("Switch model?", `Switch to ${PROVIDER_ID}/${modelId} now?`);
	if (!yes) return;
	const model = ctx.modelRegistry.find(PROVIDER_ID, modelId);
	if (!model) {
		ctx.ui.notify("Model not found in registry — pick it via /model.", "warning");
		return;
	}
	const ok = await pi.setModel(model);
	ctx.ui.notify(ok ? `Switched to ${modelId}` : "Could not switch (no API key?)", ok ? "info" : "error");
}

// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	registerLoginVehicle(pi);

	pi.registerCommand("unsloth", {
		description: "Manage Unsloth server models & settings (add/configure/apply/status/remove)",
		handler: async (_args, ctx) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("/unsloth needs an interactive UI.", "error");
				return;
			}
			const action = await ctx.ui.select("Unsloth", [
				"Add models from server",
				"Configure a model's settings",
				"Apply settings + reload model on server",
				"Server status",
				"Remove a model",
			]);
			if (!action) return;
			if (action.startsWith("Add")) await cmdAddModels(pi, ctx);
			else if (action.startsWith("Configure")) await cmdConfigure(pi, ctx);
			else if (action.startsWith("Apply")) await cmdApplyNow(ctx);
			else if (action.startsWith("Server")) await cmdStatus(ctx);
			else if (action.startsWith("Remove")) await cmdRemove(pi, ctx);
		},
	});

	// Apply load-time settings when switching to a configured model.
	pi.on("model_select", async (event, ctx) => {
		if (event.source === "restore") return;
		if (event.model.provider !== PROVIDER_ID) return;
		const cfg = readConfig();
		const provider = cfg.providers[PROVIDER_ID];
		const settings = provider?.models[event.model.id];
		if (!provider || !settings) return;
		// Fire-and-forget: loads can take minutes; notify on completion.
		void applyLoadSettings(ctx, provider, event.model.id, settings);
	});

	// Swap in thinking-specific sampling when thinking is active.
	pi.on("before_provider_request", (event, ctx) => {
		const model = ctx.model;
		if (!model || model.provider !== PROVIDER_ID) return;
		if (ctx.thinkingLevel === "off") return;
		const settings = readConfig().providers[PROVIDER_ID]?.models[model.id];
		if (!settings?.samplingThinking) return;
		const patch = buildSamplingParams(settings.samplingThinking);
		if (!patch) return;
		return { ...event.payload, ...patch };
	});

	pi.on("session_start", async (_event, ctx) => {
		sessionCtx = ctx;
		if (ctx.hasUI) ctx.ui.setWidget("unsloth-status", undefined);
	});
}
