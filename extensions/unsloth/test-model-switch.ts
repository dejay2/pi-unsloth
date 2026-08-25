/** Unit tests for Pi model-switch result handling. */
import assert from "node:assert/strict";
import { switchPiModel } from "./model-switch.ts";

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

const model = { id: "org/model:Q4" };

await test("reports a successful Pi model switch", async () => {
	let calls = 0;
	const result = await switchPiModel({
		async setModel(value) {
			calls++;
			assert.equal(value, model);
			return true;
		},
	}, model);
	assert.deepEqual(result, { ok: true });
	assert.equal(calls, 1);
});

await test("reports when Pi rejects a model switch", async () => {
	const result = await switchPiModel({ async setModel() { return false; } }, model);
	assert.deepEqual(result, { ok: false, reason: "rejected" });
});

await test("reports when Pi throws during a model switch", async () => {
	const error = new Error("missing credentials");
	const result = await switchPiModel({ async setModel() { throw error; } }, model);
	assert.equal(result.ok, false);
	if (!result.ok) {
		assert.equal(result.reason, "threw");
		assert.equal(result.error, error);
	}
});

console.log(`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`);
