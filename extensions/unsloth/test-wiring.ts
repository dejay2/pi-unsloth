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

declare const process: { exitCode?: number };

const here = dirname(fileURLToPath(import.meta.url));
const sources = readdirSync(here).filter(
	(f: string) => f.endsWith(".ts") && !f.startsWith("test-"),
);

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
	const re =
		/export\s+(?:async\s+)?(?:function|const|let|class|interface|type)\s+([A-Za-z_$][\w$]*)/g;
	let m: RegExpExecArray | null;
	while ((m = re.exec(code))) names.add(m[1] ?? "");
	return names;
}

for (const [file, code] of localModules) {
	const importRe = /import\s*(?:type\s*)?\{([^}]+)\}\s*from\s*"(\.\/[^"]+)"/g;
	let m: RegExpExecArray | null;
	while ((m = importRe.exec(code))) {
		const importPath = m[2] ?? "";
		const spec = importPath.endsWith(".ts") ? importPath : `${importPath}.ts`;
		const targetCode = localModules.get(spec);
		const names = (m[1] ?? "")
			.split(",")
			.map(
				(s) =>
					s
						.trim()
						.replace(/^type\s+/, "")
						.split(" as ")[0] ?? "".trim(),
			)
			.filter(Boolean);
		if (names.length === 0) continue;
		check(`${file} imports from ${spec}`, () => {
			assert.ok(targetCode, `${file}: cannot resolve ${spec}`);
			const exports = exportedNames(targetCode);
			for (const name of names) {
				assert.ok(
					exports.has(name),
					`${file}: "${name}" is not exported by ${spec}`,
				);
			}
		});
	}
}

const indexCode = localModules.get("./index.ts");
if (indexCode) {
	check("Pi waits for an explicit model selection before loading", () => {
		assert.doesNotMatch(
			indexCode,
			/if \(!providerConfig\.defaultModelId\)\s*setDefaultModel\(cfg, PROVIDER_ID, model\.id\)/,
		);
		assert.doesNotMatch(
			indexCode,
			/event\.source === "restore" && provider\?\.defaultModelId/,
		);
		assert.doesNotMatch(
			indexCode,
			/startupTargetId = targetId[\s\S]*?await pi\.setModel\(target\)/,
		);
		assert.match(indexCode, /"Use model now"/);
		assert.doesNotMatch(indexCode, /"Set default model"/);
		assert.match(
			indexCode,
			/async function cmdUseModel[\s\S]*?if \(ctx\.model\?\.provider === PROVIDER_ID && ctx\.model\.id === choice\)\s*\{[\s\S]*?await applyLoadSettings\(\s*ctx,\s*provider,\s*choice,\s*cfg\.providers\[PROVIDER_ID\]\.models\[choice\]\s*,?\s*\)/,
		);
		assert.match(
			indexCode,
			/const switchResult = await switchPiModel\(pi, m\)[\s\S]*?if \(!switchResult\.ok\)/,
		);
		assert.match(
			indexCode,
			/const switched = await switchPiModel\(pi, model\)[\s\S]*?if \(!switched\.ok\)/,
		);
		assert.match(
			indexCode,
			/`Server remembered: \$\{providerConfig\?\.defaultModelId/,
		);
		assert.doesNotMatch(
			indexCode,
			/`Default: \$\{providerConfig\?\.defaultModelId/,
		);
		assert.doesNotMatch(indexCode, /could not remember the default model/);
		assert.doesNotMatch(indexCode, /Manage Unsloth models, DFlash, defaults/);
		assert.match(
			indexCode,
			/const m = sessionCtx\?\.modelRegistry\.find\(PROVIDER_ID, model\.id\)[\s\S]*?const switchResult = await switchPiModel\(pi, m\)/,
		);
		assert.match(
			indexCode,
			/async function offerSwitch[\s\S]*?if \(ctx\.model\?\.provider === PROVIDER_ID && ctx\.model\.id === modelId\)[\s\S]*?await applyLoadSettings\(\s*ctx,\s*provider,\s*modelId,\s*settings\s*,?\s*\)/,
		);
		assert.match(
			indexCode,
			/async function offerSwitch[\s\S]*?const switched = await switchPiModel\(pi, model\)[\s\S]*?switched\.ok/,
		);
		assert.match(
			indexCode,
			/async function loginFlow[\s\S]*?loadModelIfNeeded\(\s*model\.id/,
		);
		assert.match(
			indexCode,
			/pi\.on\("session_start", async \(_event, ctx\) => \{[\s\S]*?sessionCtx = ctx/,
		);
		assert.doesNotMatch(
			indexCode,
			/pi\.on\("session_start", async \(_event, ctx\) => \{[\s\S]*?applySelectedModelIfNeeded/,
		);
	});
}

console.log(`\n${passed} checks passed${failed ? `, ${failed} FAILED` : ""}`);
if (failed) process.exitCode = 1;
