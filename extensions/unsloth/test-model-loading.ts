/** Regression tests for avoiding duplicate Unsloth model loads. */
import assert from "node:assert/strict";
import { fetchStatus, loadModel } from "./api.ts";
import { loadModelIfNeeded, type ModelLoadStatus } from "./model-loading.ts";

declare const process: { exitCode?: number };

let passed = 0;
async function test(name: string, fn: () => Promise<void>): Promise<void> {
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

await test("does not reload the exact model and quant already active in Unsloth", async () => {
	let loads = 0;
	const result = await loadModelIfNeeded(
		"org/model:Q4_K_M",
		async (): Promise<ModelLoadStatus> => ({
			activeModel: "org/model",
			modelIdentifier: "org/model",
			ggufVariant: "Q4_K_M",
		}),
		async () => {
			loads++;
		},
	);

	assert.equal(result, "already-loaded");
	assert.equal(loads, 0);
});

await test("loads when the active quant is different", async () => {
	let loads = 0;
	const result = await loadModelIfNeeded(
		"org/model:Q4_K_M",
		async (): Promise<ModelLoadStatus> => ({
			activeModel: "org/model",
			modelIdentifier: "org/model",
			ggufVariant: "Q8_0",
		}),
		async () => {
			loads++;
		},
	);

	assert.equal(result, "loaded");
	assert.equal(loads, 1);
});

await test("does not POST /load when the mock server already has the exact quant", async () => {
	let loadRequests = 0;
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async (input, _init) => {
		const url = String(input);
		if (url.endsWith("/api/inference/status")) {
			return new Response(
				JSON.stringify({
					active_model: "org/model",
					model_identifier: "org/model",
					gguf_variant: "Q4_K_M",
					supports_reasoning: false,
				}),
				{ status: 200, headers: { "content-type": "application/json" } },
			);
		}
		if (url.endsWith("/api/inference/load")) loadRequests++;
		return new Response("not found", { status: 404 });
	};
	try {
		const cfg = {
			baseUrl: "http://mock-unsloth/v1",
			api: "openai-completions" as const,
			apiKey: "test-key",
		};
		const result = await loadModelIfNeeded(
			"org/model:Q4_K_M",
			() => fetchStatus(cfg),
			async () => {
				await loadModel(cfg, {
					model_path: "org/model",
					gguf_variant: "Q4_K_M",
				});
			},
		);
		assert.equal(result, "already-loaded");
		assert.equal(loadRequests, 0);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

console.log(
	`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`,
);
