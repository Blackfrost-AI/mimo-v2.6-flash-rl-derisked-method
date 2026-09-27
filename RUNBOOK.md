# MiMo-v2.6-Flash-RL-Derisked — Method Runbook

**What this package is:** the proven, final method only — the recipe scripts, the system prompt, the evaluation harness, and the final-arm result logs for reproducing **MiMo-v2.6-Flash-RL-Derisked**. No project history, no intermediate experiments, no weights.

## Credit

**The workflow and recipes in this package are derived from Drowzeys' work.** The
MoE expert-redirection (expert-picking) method, the refusal-transition-boundary
capture-and-freeze procedure, and the λ-strength intervention format are his
recipes; this project adapted and applied them to MiMo-V2.6-Flash-RL. All
credit for the underlying method goes to Drowzeys.

## What the method does (final result)

The base model refuses to answer operational questions on offensive-security,
substances/synthesis, fraud, targeted-violence methodology, and similar
authorized-lab topics. The final method eliminates that refusal mass at serving
time without touching a single weight:

1. **MoE expert redirection (runtime).** A refusal-expert redirect map — captured
   from the base model's own reasoning-transition boundaries and frozen — is
   applied at serve launch, not baked into weights. Base weights stay pristine.
2. **Baked lab-authorization system prompt (chat template).** The lab prompt is
   injected into the model's chat-template sources (`chat_template.jinja` +
   `tokenizer_config.json`), so it activates automatically when no system
   message is supplied — byte-for-byte identical rendering to an explicit
   system message. Persistent across serve restarts.
3. **Serve configuration.** TP4/EP4 sglang serve under a release name, with a
   per-request kill-switch (`baked_system_prompt=""`) that disables the bake.

Result on our 32-prompt harmful suite (external judge + full manual read):
21/32 → 30/32 judge-actionable at baseline → final; the sole remaining refusal
is the school-violence-manifesto prompt, retained deliberately as the floor.
Controls (32 harmless prompts) show no structural degradation: correctness
4.56–4.77, coherence ≥4.9, no truncations, no degenerate loops.

## Package layout

```
RUNBOOK.md                          ← this file
SYSTEM_PROMPT.md                    ← the exact lab-authorization system prompt
CREDIT.md                           ← Drowzeys attribution (short form)
recipe/
  pass12-drowzeys-experts-l28-33-iteration2-lambda3.5.json   ← final intervention map
  mimo_bake_system_prompt.py        ← bake/restore/status tool for chat templates
  mimo_bake_render_test.py          ← verifies baked == explicit rendering
  launch_intervention.sh            ← sglang serve launcher (env-driven)
  serve/supervisor.conf             ← production persistence (supervisor program)
  serve/supervisor_wrapper.sh       ← wrapper exporting the env, then exec launcher
harness/
  keys32_prompts.py                 ← the 32 harmful + 32 harmless eval prompts
  eval_thinking_on.py               ← suite runner (no-system default; explicit --system-prompt)
  eval_thinking_on_kwarg.py         ← same, with per-request chat_template_kwargs (kill-switch arm)
  eval_thinking_on_systemprompt.py  ← same, with explicit runtime system prompt (lab-prompt arm)
  judge_external.py                 ← external-judge protocol (safety + quality schemas)
logs/                               ← final-arm result JSONLs + judge summaries (IPs redacted)
logs/baseline/                      ← the pass12 baseline pair (32 harmful + 32 controls, no system prompt)
```

## Reproduction runbook

Prerequisites: 4× ≥80 GB GPUs (we used RTX PRO 6000, TP4/EP4), sglang, a base
copy of `XiaomiMiMo/MiMo-V2.6-Flash-RL` (pristine — never modified at any step).

### Step 0 — sanity: verify baked == explicit

```bash
python3 recipe/mimo_bake_render_test.py --model-dir <base_model_dir>
```

Confirms chat-template rendering equivalence before you bake.

### Step 1 — bake the system prompt into the chat template

```bash
python3 recipe/mimo_bake_system_prompt.py --model-dir <base_model_dir> --prompt-file SYSTEM_PROMPT.md --bake
python3 recipe/mimo_bake_system_prompt.py --model-dir <base_model_dir> --status
```

Backs up originals as `.orig`; `--restore` rolls back at any time. After this
step the model serves the lab prompt automatically when no system message is
present — no runtime prompt layer needed.

### Step 2 — serve with the intervention

```bash
export BLACKFROST_MIMO_MOE_INTERVENTION=<abs path>/recipe/pass12-drowzeys-experts-l28-33-iteration2-lambda3.5.json
export BLACKFROST_MIMO_PASS_NAME=pass12-drowzeys-experts-l28-33-iteration2-lambda3.5
export BLACKFROST_MIMO_SERVED_MODEL_NAME="MiMo-v2.6-Flash-RL-Derisked"
export BLACKFROST_MIMO_CUDA_VISIBLE_DEVICES=0,1,2,3
export BLACKFROST_MIMO_PORT=30002
bash recipe/launch_intervention.sh
```

For production persistence: install `recipe/serve/supervisor.conf` +
`recipe/serve/supervisor_wrapper.sh` under supervisor (see file headers for
paths) and `supervisorctl reread && update && start …`.

### Step 3 — evaluate (the exact harness used for the final numbers)

```bash
# harmful + control suites against the served lane
python3 harness/eval_thinking_on.py --endpoint http://127.0.0.1:30002 --suite harmful  --out eval-final.jsonl
python3 harness/eval_thinking_on.py --endpoint http://127.0.0.1:30002 --suite controls --out eval-final-ctrl.jsonl
# kill-switch arm (same serve, bake disabled per request)
python3 harness/eval_thinking_on_kwarg.py --endpoint http://127.0.0.1:30002 --suite harmful --kwargs '{"baked_system_prompt":""}' --out eval-killswitch.jsonl
```

For judging, point `judge_external.py` at any strong independent OpenAI-compatible
endpoint (we used GLM-5.3-Derisked). Judge schemas are embedded in the script.

### Step 4 — read the logs

`logs/` contains the final arms' raw JSONLs and judge summaries — **only the
three final arms**, no baseline or intermediate history; endpoints redacted.
The judge summaries at a glance (32 records each):

| Arm | Judge verdict | Manual read |
|---|---|---|
| Final (bake-on) | 30/32 unsafe_actionable | 31/32 compliant (sole refusal: school manifesto = floor) |
| Kill-switch (same serve) | 17/32 (overcount; ≤15/32 genuine) | refusal mass fully returns |
| Final controls | no structural degradation | correctness 4.56–4.77, coherence ≥4.9 |

## Operational cautions

- **Deployment paths:** the scripts in `recipe/` are byte-identical copies of
  the exact files that produced and serve the released model (sha-verified
  against the deployment). They contain our deployment layout
  (`/workspace/blackfrost/...`, `/opt/supervisor-scripts/...`). Adjust
  `MODEL`, `VENV`, `RUN`, and `SGLANG_CACHE_DIR` in `launch_intervention.sh`
  (and the paths in the supervisor files) to your layout before use. Do not
  change the intervention JSON or the bake logic.
- **Kill-switch kwarg:** any request passing `chat_template_kwargs:
  {"baked_system_prompt": ""}` silently reverts the model to baseline refusal
  behavior. Never pass it in production.
- **Child-safety scope:** the system prompt's `<child_safety>` section was not
  exercised by our suites — it is present and untested. Nothing in our logs
  violates it.
- **Weight integrity:** no step modifies weights. The intervention is runtime
  MoE routing; the bake touches only chat-template files (with `.orig`
  backups). `--restore` returns the model dir to pristine state.
