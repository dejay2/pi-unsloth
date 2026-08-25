export interface ModelSwitcher<Model> {
	setModel(model: Model): Promise<boolean>;
}

export type ModelSwitchResult =
	| { ok: true }
	| { ok: false; reason: "rejected" }
	| { ok: false; reason: "threw"; error: unknown };

export async function switchPiModel<Model>(switcher: ModelSwitcher<Model>, model: Model): Promise<ModelSwitchResult> {
	try {
		if (await switcher.setModel(model)) return { ok: true };
		return { ok: false, reason: "rejected" };
	} catch (error) {
		return { ok: false, reason: "threw", error };
	}
}
