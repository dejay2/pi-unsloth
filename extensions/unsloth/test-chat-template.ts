/** Unit tests for Hugging Face chat-template discovery and selection helpers. */
import assert from "node:assert/strict";
import {
	fetchHuggingFaceTemplates,
	normalizeHuggingFaceModel,
	mergeTemplateLibrary,
	type ChatTemplate,
} from "./chat-template.ts";

let passed = 0;
async function test(name: string, fn: () => void | Promise<void>) {
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

await test("normalizes a Hugging Face model name or page URL", () => {
	assert.equal(normalizeHuggingFaceModel("Qwen/Qwen3-32B"), "Qwen/Qwen3-32B");
	assert.equal(normalizeHuggingFaceModel("https://huggingface.co/Qwen/Qwen3-32B/tree/main"), "Qwen/Qwen3-32B");
	assert.throws(() => normalizeHuggingFaceModel("not-a-model"), /owner\/model/);
});

await test("fetches a single tokenizer chat template as the default", async () => {
	const requested: string[] = [];
	const templates = await fetchHuggingFaceTemplates("Qwen/Qwen3-32B", async (url) => {
		requested.push(String(url));
		return new Response(JSON.stringify({ chat_template: "{% for message in messages %}ONE{% endfor %}" }), { status: 200 });
	});
	assert.deepEqual(requested, ["https://huggingface.co/Qwen/Qwen3-32B/resolve/main/tokenizer_config.json"]);
	assert.deepEqual(templates, [{
		id: "hf:Qwen/Qwen3-32B:default",
		name: "Qwen/Qwen3-32B — default",
		content: "{% for message in messages %}ONE{% endfor %}",
		source: "https://huggingface.co/Qwen/Qwen3-32B",
		isDefault: true,
	}]);
});

await test("fetches a template dictionary and marks the named default", async () => {
	const templates = await fetchHuggingFaceTemplates("org/model", async () => new Response(JSON.stringify({
		chat_template: {
			tool_use: "TOOLS",
			default: "NORMAL",
		},
	}), { status: 200 }));
	assert.deepEqual(templates.map((t) => [t.name, t.content, t.isDefault]), [
		["org/model — default", "NORMAL", true],
		["org/model — tool_use", "TOOLS", false],
	]);
});

await test("reports missing, private, and malformed Hugging Face templates", async () => {
	await assert.rejects(
		() => fetchHuggingFaceTemplates("org/private", async () => new Response("no", { status: 401 })),
		/private or requires a Hugging Face token/,
	);
	await assert.rejects(
		() => fetchHuggingFaceTemplates("org/missing", async () => new Response(JSON.stringify({}), { status: 200 })),
		/does not publish a chat template/,
	);
});

await test("merges templates by id while keeping the newest copy", () => {
	const old: ChatTemplate = { id: "x", name: "Old", content: "OLD" };
	const newer: ChatTemplate = { id: "x", name: "New", content: "NEW" };
	assert.deepEqual(mergeTemplateLibrary([old], [newer]), [newer]);
});

console.log(`\n${passed} tests passed${process.exitCode ? " (with failures)" : ""}`);
