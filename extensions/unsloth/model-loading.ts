import { splitModelRef } from "./config.ts";
import type { UnslothStatus } from "./api.ts";

export type ModelLoadStatus = Pick<
	UnslothStatus,
	"activeModel" | "modelIdentifier" | "ggufVariant"
>;

export type ModelLoadDecision = "already-loaded" | "loaded";

function statusModelRef(status: ModelLoadStatus): string | undefined {
	return status.modelIdentifier ?? status.activeModel;
}

function statusVariant(
	status: ModelLoadStatus,
	modelRef: string,
): string | undefined {
	return status.ggufVariant ?? splitModelRef(modelRef).variant;
}

export function activeModelMatches(
	status: ModelLoadStatus | null,
	modelId: string,
): boolean {
	if (!status?.activeModel) return false;
	const activeRef = statusModelRef(status);
	if (!activeRef) return false;

	const target = splitModelRef(modelId);
	const active = splitModelRef(activeRef);
	if (active.modelPath !== target.modelPath) return false;
	if (target.variant === undefined) return true;
	return statusVariant(status, activeRef) === target.variant;
}

export async function loadModelIfNeeded(
	modelId: string,
	readStatus: () => Promise<ModelLoadStatus | null>,
	load: () => Promise<void>,
): Promise<ModelLoadDecision> {
	if (activeModelMatches(await readStatus(), modelId)) return "already-loaded";
	await load();
	return "loaded";
}
