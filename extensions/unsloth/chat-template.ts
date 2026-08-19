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

function templatesFromTokenizer(model: string, raw: unknown): ChatTemplate[] {
	const source = `https://huggingface.co/${model}`;
	const entries: Array<[string, string]> = typeof raw === "string"
		? [["default", raw]]
		: raw && typeof raw === "object" && !Array.isArray(raw)
			? Object.entries(raw).filter((entry): entry is [string, string] => typeof entry[1] === "string")
			: [];
	entries.sort(([a], [b]) => (a === "default" ? -1 : b === "default" ? 1 : a.localeCompare(b)));
	return entries.map(([key, content], index) => ({
		id: `hf:${model}:${key}`,
		name: `${model} — ${key}`,
		content,
		source,
		isDefault: key === "default" || (entries.length === 1 && index === 0),
	}));
}

function baseModelFromMetadata(body: any): string | undefined {
	const raw = body?.cardData?.base_model ?? body?.base_model;
	const values = Array.isArray(raw) ? raw : [raw];
	return values.find((value): value is string => typeof value === "string" && /^[^/\s]+\/[^/\s]+$/.test(value));
}

export async function fetchHuggingFaceTemplates(
	rawModel: string,
	fetcher: typeof fetch = fetch,
): Promise<ChatTemplate[]> {
	const firstModel = normalizeHuggingFaceModel(rawModel);
	const visited = new Set<string>();

	async function find(model: string, depth: number): Promise<ChatTemplate[]> {
		if (depth > 4 || visited.has(model)) throw new Error("Hugging Face model links form a loop or are too deeply nested");
		visited.add(model);
		const source = `https://huggingface.co/${model}`;
		const tokenizerResponse = await fetcher(`${source}/resolve/main/tokenizer_config.json`, {
			headers: { accept: "application/json" },
		});
		if (tokenizerResponse.status === 401 || tokenizerResponse.status === 403) {
			throw new Error("This Hugging Face model is private or requires a Hugging Face token");
		}
		if (tokenizerResponse.ok) {
			try {
				const body: any = await tokenizerResponse.json();
				const templates = templatesFromTokenizer(model, body?.chat_template);
				if (templates.length > 0) return templates;
			} catch {
				throw new Error("Hugging Face returned an invalid tokenizer configuration");
			}
		} else if (tokenizerResponse.status !== 404) {
			throw new Error(`Could not read the Hugging Face model (HTTP ${tokenizerResponse.status})`);
		}

		const metadataResponse = await fetcher(`https://huggingface.co/api/models/${model}`, {
			headers: { accept: "application/json" },
		});
		if (metadataResponse.status === 401 || metadataResponse.status === 403) {
			throw new Error("This Hugging Face model is private or requires a Hugging Face token");
		}
		if (!metadataResponse.ok) throw new Error(`Could not read the Hugging Face model (HTTP ${metadataResponse.status})`);
		let metadata: any;
		try {
			metadata = await metadataResponse.json();
		} catch {
			throw new Error("Hugging Face returned invalid model information");
		}
		const files = Array.isArray(metadata?.siblings)
			? metadata.siblings.map((item: any) => item?.rfilename).filter((name: unknown): name is string => typeof name === "string")
			: [];
		if (files.includes("chat_template.jinja")) {
			const templateResponse = await fetcher(`${source}/resolve/main/chat_template.jinja`);
			if (templateResponse.ok) {
				const content = (await templateResponse.text()).trim();
				if (content) return [{
					id: `hf:${model}:chat_template`,
					name: `${model} — chat_template`,
					content,
					source,
					isDefault: true,
				}];
			}
		}
		const baseModel = baseModelFromMetadata(metadata);
		if (baseModel) return find(baseModel, depth + 1);
		throw new Error("This Hugging Face model does not publish a chat template or link to a parent model that does");
	}

	return find(firstModel, 0);
}

export function mergeTemplateLibrary(...groups: ChatTemplate[][]): ChatTemplate[] {
	const byId = new Map<string, ChatTemplate>();
	for (const group of groups) for (const template of group) byId.set(template.id, template);
	return [...byId.values()];
}
