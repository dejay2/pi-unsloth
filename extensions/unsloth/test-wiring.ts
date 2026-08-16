/**
 * Wiring test: verifies every named import from local modules actually
 * exists as an export in the target file. Catches the "reference without
 * import" / "import never added" class of error that pi's load check
 * cannot surface (wizard code only runs interactively).
 *
 * Run with: node test-wiring.ts
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const sources = readdirSync(here).filter((f) => f.endsWith(".ts") && !f.startsWith("test-"));

let passed = 0;
let failed = 0;

function check(name: string, fn: () => void) {
	try {
		fn();
		passed++;
		console.log(`ok   ${name}`);
	} catch (err) {
		failed++;
		console.error(`FAIL ${name}`);
		console.error(err instanceof Error ? err.message : err);
	}
}

const localModules = new Map<string, string>();
for (const f of sources) {
	localModules.set(`./${f}`, readFileSync(join(here, f), "utf8"));
}

function exportedNames(code: string): Set<string> {
	const names = new Set<string>();
	const re = /export\s+(?:async\s+)?(?:function|const|let|class|interface|type)\s+([A-Za-z_$][\w$]*)/g;
	let m;
	while ((m = re.exec(code))) names.add(m[1]);
	return names;
}

for (const [file, code] of localModules) {
	const importRe = /import\s*(?:type\s*)?\{([^}]+)\}\s*from\s*"(\.\/[^"]+)"/g;
	let m;
	while ((m = importRe.exec(code))) {
		const spec = m[2].endsWith(".ts") ? m[2] : `${m[2]}.ts`;
		const targetCode = localModules.get(spec);
		const names = m[1]
			.split(",")
			.map((s) => s.trim().replace(/^type\s+/, "").split(" as ")[0].trim())
			.filter(Boolean);
		if (names.length === 0) continue;
		check(`${file} imports from ${spec}`, () => {
			assert.ok(targetCode, `${file}: cannot resolve ${spec}`);
			const exports = exportedNames(targetCode);
			for (const name of names) {
				assert.ok(exports.has(name), `${file}: "${name}" is not exported by ${spec}`);
			}
		});
	}
}

console.log(`\n${passed} checks passed${failed ? `, ${failed} FAILED` : ""}`);
if (failed) process.exitCode = 1;
