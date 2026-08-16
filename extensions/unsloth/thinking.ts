/**
 * Thinking/reasoning control mapping for custom endpoints.
 *
 * Unsloth Studio classifies the loaded model's chat template (GGUF sniffer)
 * and exposes the result at GET /api/inference/status:
 *   - reasoning_style: "enable_thinking" (Qwen3-style on/off),
 *     "enable_thinking_effort" (gate + discrete levels, e.g. GLM high|max),
 *     "reasoning_effort" (levels only, e.g. gpt-oss low|medium|high)
 *   - reasoning_effort_levels: the levels the template actually branches on
 *   - reasoning_always_on: no off switch exists
 *
 * thinkingConfigFor() maps that classification to pi models.json config:
 *   - enable_thinking        → compat.thinkingFormat "qwen-chat-template"
 *                              (pi sends chat_template_kwargs.enable_thinking
 *                              true/false + preserve_thinking; Unsloth lifts it)
 *   - enable_thinking_effort → compat.thinkingFormat "chat-template" sending
 *                              both enable_thinking and reasoning_effort kwargs,
 *                              with thinkingLevelMap pinning pi levels to the
 *                              template's levels (unsupported levels hidden)
 *   - reasoning_effort       → chat-template sending reasoning_effort, with
 *                              thinkingLevelMap off→"none" (llama.cpp's disable
 *                              sentinel for these templates)
 *   - always-on              → thinkingLevelMap { off: null } (pi hides "off")
 *
 * No pi imports — unit-testable (test-thinking.ts).
 */

import { serverRoot, type DiscoveryConfig } from "./discover.ts";

export type ReasoningStyle = "enable_thinking" | "enable_thinking_effort" | "reasoning_effort";

export interface ReasoningInfo {
	style: ReasoningStyle | "always_on" | "none";
	levels: string[];
	supportsPreserveThinking: boolean;
	/** Display id of the currently-loaded model this classification describes. */
	activeModel?: string;
}

/** pi thinking levels that can be mapped to provider values. */
const PI_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;

export interface ThinkingConfig {
	reasoning: boolean;
	compat?: Record<string, unknown>;
	thinkingLevelMap?: Record<string, string | null>;
}

/** Fetch Unsloth's reasoning classification for the loaded model. null when not Unsloth. */
export async function fetchUnslothReasoning(cfg: DiscoveryConfig, timeoutMs = 8_000): Promise<ReasoningInfo | null> {
	const url = `${serverRoot(cfg.baseUrl)}/api/inference/status`;
	try {
		const headers: Record<string, string> = { accept: "application/json" };
		if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`;
		const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
		if (!res.ok) return null;
		const body = (await res.json()) as Record<string, unknown>;
		if (typeof body.supports_reasoning !== "boolean") return null; // not an Unsloth status payload
		const activeModel = typeof body.active_model === "string" ? body.active_model : undefined;
		if (!body.supports_reasoning) {
			return { style: "none", levels: [], supportsPreserveThinking: false, activeModel };
		}
		if (body.reasoning_always_on === true) {
			return { style: "always_on", levels: [], supportsPreserveThinking: false, activeModel };
		}
		const style = body.reasoning_style;
		const levels = Array.isArray(body.reasoning_effort_levels)
			? body.reasoning_effort_levels.filter((l): l is string => typeof l === "string")
			: [];
		if (style === "enable_thinking" || style === "enable_thinking_effort" || style === "reasoning_effort") {
			return {
				style,
				levels,
				supportsPreserveThinking: body.supports_preserve_thinking === true,
				activeModel,
			};
		}
		return null;
	} catch {
		return null;
	}
}

function mapLevels(levels: string[]): Record<string, string | null> {
	const map: Record<string, string | null> = {};
	for (const l of PI_LEVELS) map[l] = levels.includes(l) ? l : null;
	return map;
}

const DEFAULT_EFFORT_LEVELS = ["low", "medium", "high"];

/** Map a detected reasoning style to pi model config. */
export function thinkingConfigFor(info: ReasoningInfo): ThinkingConfig {
	switch (info.style) {
		case "enable_thinking":
			return {
				reasoning: true,
				compat: {
					thinkingFormat: "qwen-chat-template",
					supportsReasoningEffort: false,
					supportsDeveloperRole: false,
				},
			};
		case "enable_thinking_effort":
			return {
				reasoning: true,
				thinkingLevelMap: mapLevels(info.levels.length > 0 ? info.levels : DEFAULT_EFFORT_LEVELS),
				compat: {
					thinkingFormat: "chat-template",
					chatTemplateKwargs: {
						enable_thinking: { $var: "thinking.enabled" },
						reasoning_effort: { $var: "thinking.effort" },
					},
					supportsReasoningEffort: false,
					supportsDeveloperRole: false,
				},
			};
		case "reasoning_effort":
			return {
				reasoning: true,
				thinkingLevelMap: {
					off: "none", // llama.cpp disable sentinel for effort-style templates
					...mapLevels(info.levels.length > 0 ? info.levels : DEFAULT_EFFORT_LEVELS),
				},
				compat: {
					thinkingFormat: "chat-template",
					chatTemplateKwargs: {
						reasoning_effort: { $var: "thinking.effort" },
					},
					supportsReasoningEffort: false,
					supportsDeveloperRole: false,
				},
			};
		case "always_on":
			return { reasoning: true, thinkingLevelMap: { off: null } };
		default:
			return { reasoning: false };
	}
}

/**
 * Name-family fallback for models that are NOT currently loaded (so the
 * status probe says nothing about them). Returns undefined when the family
 * is unknown — callers should ask the user instead.
 */
export function guessThinkingConfig(modelId: string): ThinkingConfig | undefined {
	const id = modelId.toLowerCase();
	if (/qwen3|qwen-3/.test(id)) {
		return thinkingConfigFor({ style: "enable_thinking", levels: [], supportsPreserveThinking: true });
	}
	if (id.includes("gpt-oss")) {
		return thinkingConfigFor({ style: "reasoning_effort", levels: DEFAULT_EFFORT_LEVELS, supportsPreserveThinking: false });
	}
	if (/glm-5|glm5/.test(id)) {
		return thinkingConfigFor({ style: "enable_thinking_effort", levels: ["high", "max"], supportsPreserveThinking: false });
	}
	return undefined;
}

/** Loose match: does this picked model id refer to the actively-loaded model? */
export function matchesActiveModel(pickedId: string, activeModel?: string): boolean {
	if (!activeModel) return false;
	const base = pickedId.split(":")[0].toLowerCase();
	const active = activeModel.toLowerCase();
	if (base === active) return true;
	// tolerate basename vs repo-id forms
	const baseName = base.split("/").pop() ?? base;
	const activeName = active.split("/").pop() ?? active;
	return baseName === activeName;
}
