/**
 * Unit tests for api.ts against a mock Unsloth server — run with: node test-api.ts
 */
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { fetchStatus, listModels, loadModel } from "./api.ts";

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

let server: Server;
let lastLoadBody: any;
let lastAuth: string | undefined;

function cfgFor(port: number) {
	return { baseUrl: `http://127.0.0.1:${port}/v1`, api: "openai-completions", apiKey: "sk-u" };
}

await test("status, model list (with quant expansion), and load round-trip", async () => {
	server = createServer((req, res) => {
		lastAuth = req.headers.authorization;
		let body = "";
		req.on("data", (c) => (body += c));
		req.on("end", () => {
			const u = new URL(req.url ?? "", "http://x");
			if (u.pathname === "/api/inference/status") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({
					active_model: "org/m",
					loaded: ["org/m"],
					supports_reasoning: true,
					reasoning_style: "enable_thinking",
					reasoning_effort_levels: [],
					reasoning_always_on: false,
				}));
			} else if (u.pathname === "/v1/models") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ data: [{ id: "org/m", quant: "Q4", loaded: true, context_length: 64000 }] }));
			} else if (u.pathname === "/api/models/gguf-variants") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({
					variants: [
						{ quant: "Q4", size_bytes: 4e9, downloaded: true },
						{ quant: "Q8", size_bytes: 8e9, downloaded: true },
					],
				}));
			} else if (u.pathname === "/api/inference/load" && req.method === "POST") {
				lastLoadBody = JSON.parse(body);
				res.writeHead(200, { "content-type": "application/json" });
				res.end("{}");
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
	assert.equal(status?.reasoning?.style, "enable_thinking");
	assert.equal(lastAuth, "Bearer sk-u");

	// model list with per-quant expansion
	const models = await listModels(cfg);
	assert.deepEqual(models?.map((m) => m.id), ["org/m:Q4", "org/m:Q8"]);
	assert.equal(models?.[0].contextWindow, 64000);

	// load
	const res = await loadModel(cfg, { model_path: "org/m", gguf_variant: "Q8", speculative_type: "off" });
	assert.equal(res.ok, true);
	assert.deepEqual(lastLoadBody, { model_path: "org/m", gguf_variant: "Q8", speculative_type: "off" });

	await new Promise<void>((r) => server.close(() => r()));
});

await test("loadModel surfaces 409 + detail", async () => {
	const s = createServer((_req, res) => {
		res.writeHead(409, { "content-type": "application/json" });
		res.end(JSON.stringify({ detail: "active requests in flight" }));
	});
	await new Promise<void>((r) => s.listen(0, r));
	const res = await loadModel(cfgFor((s.address() as AddressInfo).port), { model_path: "x" });
	assert.equal(res.ok, false);
	assert.equal(res.status, 409);
	assert.ok(res.error?.includes("active requests"));
	await new Promise<void>((r) => s.close(() => r()));
});

await test("fetchStatus returns null for non-Unsloth servers", async () => {
	const s = createServer((_req, res) => {
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ hello: "world" }));
	});
	await new Promise<void>((r) => s.listen(0, r));
	assert.equal(await fetchStatus(cfgFor((s.address() as AddressInfo).port)), null);
	await new Promise<void>((r) => s.close(() => r()));
});

await test("loadModel handles connection refused without throwing", async () => {
	const res = await loadModel(cfgFor(1), { model_path: "x" }, 2000);
	assert.equal(res.ok, false);
	assert.equal(res.status, 0);
	assert.ok(res.error);
});

console.log(`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`);
