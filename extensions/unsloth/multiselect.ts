/**
 * MultiSelect — a checkbox-style list for scoping which discovered models to add.
 *
 * Keys: ↑/↓ (or k/j) navigate • space toggle • a all/none • enter confirm • esc cancel.
 * Enter with nothing selected selects the item under the cursor.
 *
 * Rendering is theme-injected so the interaction logic is unit-testable.
 */

export interface MultiSelectTheme {
	accent(s: string): string;
	muted(s: string): string;
	dim(s: string): string;
	bold(s: string): string;
	warning(s: string): string;
}

/**
 * Key matcher, e.g. `matchesKey` from @earendil-works/pi-tui. Injected so the
 * component stays dependency-free and unit-testable — raw terminal input
 * differs between legacy escape sequences and the Kitty keyboard protocol,
 * and the injected matcher normalizes both.
 */
export type KeyMatcher = (data: string, keyId: string) => boolean;

export class MultiSelect {
	private cursor = 0;
	private readonly selected = new Set<number>();
	private offset = 0;
	private readonly items: string[];
	private readonly maxVisible: number;
	private readonly theme: MultiSelectTheme;
	private readonly done: (result: string[] | null) => void;
	private readonly match: KeyMatcher;

	constructor(
		items: string[],
		maxVisible: number,
		theme: MultiSelectTheme,
		done: (result: string[] | null) => void,
		match?: KeyMatcher,
		preselected?: Iterable<number>,
	) {
		this.items = items;
		this.maxVisible = maxVisible;
		this.theme = theme;
		this.done = done;
		if (preselected) for (const i of preselected) this.selected.add(i);
		// Fallback for tests: legacy escape sequences + raw printable chars.
		this.match =
			match ??
			((data, keyId) => {
				const legacy: Record<string, string> = {
					up: "\x1b[A",
					down: "\x1b[B",
					escape: "\x1b",
					enter: "\r",
					space: " ",
				};
				return data === (legacy[keyId] ?? keyId);
			});
	}

	/** Current selection as item values (for tests). */
	get selectedItems(): string[] {
		return this.items.filter((_, i) => this.selected.has(i));
	}

	handleInput(data: string): void {
		if (this.match(data, "escape")) {
			this.done(null);
			return;
		}
		if (this.match(data, "up") || this.match(data, "k")) {
			this.move(-1);
			return;
		}
		if (this.match(data, "down") || this.match(data, "j")) {
			this.move(1);
			return;
		}
		if (this.match(data, "space")) {
			this.toggle(this.cursor);
			return;
		}
		if (this.match(data, "a")) {
			if (this.selected.size === this.items.length) this.selected.clear();
			else this.items.forEach((_, i) => this.selected.add(i));
			return;
		}
		if (this.match(data, "enter")) {
			if (this.selected.size === 0) this.selected.add(this.cursor);
			this.done(this.selectedItems);
			return;
		}
	}

	private move(delta: number): void {
		const n = this.items.length;
		if (n === 0) return;
		this.cursor = (this.cursor + delta + n) % n;
		if (this.cursor < this.offset) this.offset = this.cursor;
		if (this.cursor >= this.offset + this.maxVisible) this.offset = this.cursor - this.maxVisible + 1;
	}

	private toggle(index: number): void {
		if (this.selected.has(index)) this.selected.delete(index);
		else this.selected.add(index);
	}

	render(_width: number): string[] {
		const t = this.theme;
		const lines: string[] = [];
		lines.push(t.accent(t.bold(`Select models to add (${this.selected.size}/${this.items.length} chosen)`)));
		lines.push("");

		if (this.items.length === 0) {
			lines.push(t.warning("  (no models)"));
			return lines;
		}

		const end = Math.min(this.offset + this.maxVisible, this.items.length);
		for (let i = this.offset; i < end; i++) {
			const isCursor = i === this.cursor;
			const box = this.selected.has(i) ? "[●]" : "[ ]";
			const pointer = isCursor ? "❯" : " ";
			const label = `${pointer} ${box} ${this.items[i]}`;
			lines.push(isCursor ? t.accent(label) : label);
		}
		if (this.offset > 0 || end < this.items.length) {
			lines.push(t.dim(`  (${this.offset + 1}–${end} of ${this.items.length})`));
		}

		lines.push("");
		lines.push(t.muted("↑↓ navigate • space toggle • a all/none • enter confirm • esc cancel"));
		return lines;
	}

	invalidate(): void {}
}
