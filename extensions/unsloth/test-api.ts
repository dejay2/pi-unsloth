/**
 * Unit tests for api.ts against a mock Unsloth server — run with: node test-api.ts
 */
import assert from "node:assert/strict";
import {
	createServer,
	type IncomingMessage,
	type Server,
	type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import {
	fetchStatus,
	listModels,
	loadModel,
	saveModelOverride,
	saveLastLocalModel,
	fetchAutoSwitch,
	setAutoSwitch,
} from "./api.ts";

declare const process: { exitCode?: number };

let passed = 0;
async function test(name: string, fn: () => Promise<void>) {
	try {
		await fn();
		passed++;
		console.log(`ok   ${name}`);
	} catch (err) {
		console.error(`FAIL ${name}`);
		console.error(err);
		process.exitCode = 1;
	}
}

type JsonObject = Record<string, unknown>;

let server: Server;
let lastLoadBody: JsonObject = {};
let lastOverrideBody: JsonObject = {};
let lastLocalModelBody: JsonObject = {};
let lastAutoSwitchBody: JsonObject = {};
let lastAuth: string | undefined;

function cfgFor(port: number) {
	return {
		baseUrl: `http://127.0.0.1:${port}/v1`,
		api: "openai-completions",
		apiKey: "sk-u",
	};
}

function isJsonObject(value: unknown): value is JsonObject {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseJsonBody(body: string): JsonObject {
	let value: unknown;
	try {
		value = JSON.parse(body);
	} catch {
		throw new Error("Mock server received invalid JSON");
	}
	if (!isJsonObject(value))
		throw new Error("Mock server expected a JSON object");
	return value;
}

await test("status, model list (with quant expansion), and load round-trip", async () => {
	server = createServer((req: IncomingMessage, res: ServerResponse) => {
		lastAuth = req.headers.authorization;
		let body = "";
		req.on("data", (c: string | Uint8Array) => {
			body += typeof c === "string" ? c : new TextDecoder().decode(c);
		});
		req.on("end", () => {
			const u = new URL(req.url ?? "", "http://x");
			if (u.pathname === "/api/inference/status") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						active_model: "org/m",
						model_identifier: "org/m",
						gguf_variant: "Q4",
						loaded: ["org/m"],
						supports_reasoning: true,
						reasoning_style: "enable_thinking",
						reasoning_effort_levels: [],
						reasoning_always_on: false,
						chat_template: "BUILT_IN",
						chat_template_override: "CUSTOM",
						spec_fallback_reason: "runtime_error",
					}),
				);
			} else if (u.pathname === "/v1/models") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						data: [{ id: "org/m", quant: "Q4", loaded: true, context_length: 64000 }],
					}),
				);
			} else if (u.pathname === "/api/models/gguf-variants") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						variants: [
							{ quant: "Q4", size_bytes: 4e9, downloaded: true },
							{ quant: "Q8", size_bytes: 8e9, downloaded: true },
						],
					}),
				);
			} else if (u.pathname === "/api/inference/load" && req.method === "POST") {
				lastLoadBody = parseJsonBody(body);
				res.writeHead(200, { "content-type": "application/json" });
				res.end("{}");
			} else if (
				u.pathname === "/api/settings/openai-auto-switch/overrides" &&
				req.method === "PUT"
			) {
				lastOverrideBody = parseJsonBody(body);
				const modelId = lastOverrideBody.model_id;
				if (typeof modelId !== "string")
					throw new Error("Mock server expected model_id");
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ overrides: { [modelId]: lastOverrideBody } }));
			} else if (
				u.pathname === "/api/settings/last-local-model" &&
				req.method === "PUT"
			) {
				lastLocalModelBody = parseJsonBody(body);
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify(lastLocalModelBody));
			} else if (
				u.pathname === "/api/settings/openai-auto-switch" &&
				req.method === "GET"
			) {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						enabled: true,
						auto_unload_idle_seconds: 600,
						auto_download_model: false,
					}),
				);
			} else if (
				u.pathname === "/api/settings/openai-auto-switch" &&
				req.method === "PUT"
			) {
				lastAutoSwitchBody = parseJsonBody(body);
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify(lastAutoSwitchBody));
			} else {
				res.writeHead(404).end();
			}
		});
	});
	await new Promise<void>((r) => server.listen(0, r));
	const port = (server.address() as AddressInfo).port;
	const cfg = cfgFor(port);

	// status
	const status = await fetchStatus(cfg);
	assert.equal(status?.activeModel, "org/m");
	assert.equal(status?.modelIdentifier, "org/m");
	assert.equal(status?.ggufVariant, "Q4");
	assert.equal(status?.reasoning?.style, "enable_thinking");
	assert.equal(status?.chatTemplate, "BUILT_IN");
	assert.equal(status?.chatTemplateOverride, "CUSTOM");
	assert.equal(status?.specFallbackReason, "runtime_error");
	assert.equal(lastAuth, "Bearer sk-u");

	// model list with per-quant expansion
	const models = await listModels(cfg);
	assert.deepEqual(
		models?.map((m) => m.id),
		["org/m:Q4", "org/m:Q8"],
	);
	assert.equal(models?.[0].contextWindow, 64000);

	// load
	const res = await loadModel(cfg, {
		model_path: "org/m",
		gguf_variant: "Q8",
		speculative_type: "off",
	});
	assert.equal(res.ok, true);
	assert.deepEqual(lastLoadBody, {
		model_path: "org/m",
		gguf_variant: "Q8",
		speculative_type: "off",
	});

	const override = await saveModelOverride(cfg, "org/m:Q8", {
		load: { nParallel: 2, nBatch: 4096 },
	});
	assert.equal(override.ok, true);
	assert.deepEqual(lastOverrideBody, {
		model_id: "org/m:Q8",
		remove: false,
		llama_extra_args: [],
		n_parallel: 2,
		n_batch: 4096,
	});
	assert.equal(lastAuth, "Bearer sk-u");

	const savedDefault = await saveLastLocalModel(cfg, "org/m:Q8", 123456789);
	assert.equal(savedDefault.ok, true);
	assert.deepEqual(lastLocalModelBody, {
		id: "org/m",
		kind: "gguf",
		gguf_variant: "Q8",
		loaded_at: 123456789,
		client_now: 123456789,
	});

	const autoSwitch = await fetchAutoSwitch(cfg);
	assert.equal(autoSwitch?.enabled, true);
	const changedAutoSwitch = await setAutoSwitch(cfg, false);
	assert.equal(changedAutoSwitch.ok, true);
	assert.deepEqual(lastAutoSwitchBody, {
		enabled: false,
		auto_unload_idle_seconds: 600,
		auto_download_model: false,
	});

	await new Promise<void>((r) => server.close(() => r()));
});

await test("loadModel surfaces 409 + detail", async () => {
	const s = createServer((_req: IncomingMessage, res: ServerResponse) => {
		res.writeHead(409, { "content-type": "application/json" });
		res.end(JSON.stringify({ detail: "active requests in flight" }));
	});
	await new Promise<void>((r) => s.listen(0, r));
	const res = await loadModel(cfgFor((s.address() as AddressInfo).port), {
		model_path: "x",
	});
	assert.equal(res.ok, false);
	assert.equal(res.status, 409);
	assert.ok(res.error?.includes("active requests"));
	await new Promise<void>((r) => s.close(() => r()));
});

await test("fetchStatus returns null for non-Unsloth servers", async () => {
	const s = createServer((_req: IncomingMessage, res: ServerResponse) => {
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ hello: "world" }));
	});
	await new Promise<void>((r) => s.listen(0, r));
	assert.equal(
		await fetchStatus(cfgFor((s.address() as AddressInfo).port)),
		null,
	);
	await new Promise<void>((r) => s.close(() => r()));
});

await test("loadModel handles connection refused without throwing", async () => {
	const res = await loadModel(cfgFor(1), { model_path: "x" }, 2000);
	assert.equal(res.ok, false);
	assert.equal(res.status, 0);
	assert.ok(res.error);
});

console.log(
	`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`,
);
