/** Chat-template discovery and library helpers. */

export interface ChatTemplate {
	id: string;
	name: string;
	content: string;
	source?: string;
	isDefault?: boolean;
}

export function normalizeHuggingFaceModel(raw: string): string {
	let value = raw.trim().replace(/\/+$/, "");
	try {
		const url = new URL(value);
		if (url.hostname !== "huggingface.co" && url.hostname !== "www.huggingface.co") throw new Error();
		value = url.pathname.split("/").filter(Boolean).slice(0, 2).join("/");
	} catch {
		// A plain owner/model identifier is also accepted.
	}
	if (!/^[^/\s]+\/[^/\s]+$/.test(value)) {
		throw new Error("Enter a Hugging Face model as owner/model or paste its page address");
	}
	return value;
}

export async function fetchHuggingFaceTemplates(
	rawModel: string,
	fetcher: typeof fetch = fetch,
): Promise<ChatTemplate[]> {
	const model = normalizeHuggingFaceModel(rawModel);
	const source = `https://huggingface.co/${model}`;
	const url = `${source}/resolve/main/tokenizer_config.json`;
	const response = await fetcher(url, { headers: { accept: "application/json" } });
	if (response.status === 401 || response.status === 403) {
		throw new Error("This Hugging Face model is private or requires a Hugging Face token");
	}
	if (!response.ok) throw new Error(`Could not read the Hugging Face model (HTTP ${response.status})`);
	let body: any;
	try {
		body = await response.json();
	} catch {
		throw new Error("Hugging Face returned an invalid tokenizer configuration");
	}
	const raw = body?.chat_template;
	const entries: Array<[string, string]> = typeof raw === "string"
		? [["default", raw]]
		: raw && typeof raw === "object" && !Array.isArray(raw)
			? Object.entries(raw).filter((entry): entry is [string, string] => typeof entry[1] === "string")
			: [];
	if (entries.length === 0) throw new Error("This Hugging Face model does not publish a chat template");
	entries.sort(([a], [b]) => (a === "default" ? -1 : b === "default" ? 1 : a.localeCompare(b)));
	return entries.map(([key, content], index) => ({
		id: `hf:${model}:${key}`,
		name: `${model} — ${key}`,
		content,
		source,
		isDefault: key === "default" || (entries.length === 1 && index === 0),
	}));
}

export function mergeTemplateLibrary(...groups: ChatTemplate[][]): ChatTemplate[] {
	const byId = new Map<string, ChatTemplate>();
	for (const group of groups) for (const template of group) byId.set(template.id, template);
	return [...byId.values()];
}
