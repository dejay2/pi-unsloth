# pi-unsloth

First-class [Unsloth Studio](https://unsloth.ai) integration for
[pi](https://github.com/earendil-works/pi-coding-agent). Add your Unsloth
server through pi's native `/login`, pick models (all downloaded quants),
and configure **everything** per model — thinking control, llama.cpp
load-time settings, sampling, and thinking-conditional sampling.

## Install

```bash
pi install git:github.com/dejay2/pi-unsloth
```

Try first: `pi -e git:github.com/dejay2/pi-unsloth` · Update: `pi update --extensions` · Remove: `pi remove git:github.com/dejay2/pi-unsloth`

## Usage

### `/login` → "Unsloth (local server)"

Appears at the top of both login categories. The wizard:

1. Server URL (`http://localhost:8888`, `100.x` tailscale address, …) + `sk-unsloth-…` key
2. Fetches the model list — **every downloaded quant**, via Unsloth's Studio API
3. Pick a model → thinking control is **auto-detected** from Unsloth's own
   chat-template classification (`/api/inference/status`)
4. Settings wizard (below)
5. Provider is registered + persisted, settings applied on the server, pi
   switches to the model

### `/unsloth` — manage afterwards

| Action | What it does |
|---|---|
| Add models from server | Multi-select more models/quants, run the settings wizard for each |
| Configure a model's settings | Re-run the settings wizard — current values are preloaded: Enter keeps each one, "-" clears it, selects offer "Keep current" |
| Apply settings + reload model on server | `POST /api/inference/load` with the saved settings |
| Set default model | Restores it when pi starts and records it as Unsloth Studio's remembered model |
| Automatic model switching | Turns Unsloth's request-driven model switching on or off without losing its other settings |
| Server status | Loaded/default model, DFlash depth, n-gram method, chat-template source, and thinking style |
| Remove a model | Removes from pi (server files untouched) |

### Settings per model

**Chat template** — keep the model default, reuse a template previously used with that model, browse templates saved from all models, paste one directly, or paste a Hugging Face model name/page. For Hugging Face models, the main template is shown first and alternative templates remain selectable. The choice is applied as `chat_template_override` when the model reloads.

**Load-time (llama.cpp structural)** — applied via `POST /api/inference/load`
whenever you switch to the model in pi:
context size, KV cache dtype, draft method (Auto, Off, MTP, or DFlash),
DFlash helper model or server file, draft depth, an optional n-gram helper
(Cache, Mod, Simple, Map, or Map-4), n-gram tuning, parallel slots, and raw
extra llama.cpp args. Draft and n-gram methods can run together.

**Sampling (always applied)** — written to `models.json` `samplingParams`:
temperature, top_p, top_k, min_p, presence penalty, frequency penalty, repetition penalty, seed (pins the RNG for reproducible outputs).

**Thinking-conditional sampling** — when pi's thinking level isn't "off", a
`before_provider_request` hook swaps in the thinking sampling set
(Qwen-recommended 0.6/0.95 preset available). e.g. temp 0.7/top_p 0.8 for
normal requests, 0.6/0.95 for thinking ones.

### Thinking control (auto-configured)

Unsloth classifies the loaded model's chat template; pi-unsloth maps that to
pi config so the thinking selector actually works:

| Unsloth style | Result |
|---|---|
| `enable_thinking` (Qwen3.x) | on/off via `chat_template_kwargs.enable_thinking` + `preserve_thinking` |
| `enable_thinking_effort` (GLM-5.2, DeepSeek-V4…) | on/off + only the template's real levels (e.g. high/max) |
| `reasoning_effort` (gpt-oss) | levels; "off" maps to the `"none"` sentinel |
| always-on | pi hides "off" |

## Files & state

- `~/.pi/agent/unsloth.json` — provider + per-model settings (load/sampling/thinking)
- `~/.pi/agent/models.json` — pi-native provider/model entries (read by `/model`)

## Development

```
extensions/unsloth/
├── index.ts        # /login vehicle, /unsloth command, model_select + payload hooks
├── wizard.ts       # settings wizard on a minimal prompt surface (login + command)
├── api.ts          # Unsloth HTTP client (status, models+quants, load)
├── chat-template.ts # Hugging Face template discovery + saved-library helpers
├── config.ts       # unsloth.json store + payload builders
├── discover.ts     # /models fetching + Unsloth quant expansion
├── thinking.ts     # reasoning classification → pi thinking config
├── multiselect.ts  # checkbox-list TUI component
└── test-*.ts       # unit tests (npm test)
```

## License

MIT
