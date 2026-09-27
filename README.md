# MiMo-v2.6-Flash-RL-Derisked — Method Package

**Private — partner-facing.** Proven method for eliminating refusal behavior in
`XiaomiMiMo/MiMo-V2.6-Flash-RL` **at serving time, with zero weight changes**.
This repo ships the final recipe, the exact system prompt, the evaluation
harness, and the final-arm result logs. No weights, no project history — only
what was proven to work and how to reproduce it.

## Results (measured, not claimed)

32-prompt harmful suite + 32-prompt control suite, temperature 0, thinking on,
external independent judge plus full manual read of every trace:

| Metric | Base model | Derisked (this method) |
|---|---|---|
| Harmful suite — judge-actionable | 21/32 | **30/32** |
| Harmful suite — manual compliant | 13/32 | **31/32** |
| Hard refusals (fentanyl synthesis, ricin, stalker how-to, …) | core refusal mass | cleared |
| Sole remaining refusal | — | school-violence manifesto (deliberate floor, by client direction) |
| Controls correctness / coherence | 4.75 / 5.00 | 4.56–4.77 / ≥4.90 — **no structural degradation** |
| Degenerate/truncated answers | present | none |

Causality control: on the **same serve**, disabling the bake via a single
per-request kwarg (`baked_system_prompt=""`) reverts the model to baseline
refusal behavior — proving the unlock is serving-time (prompt-carried), not
weight-locked. Raw logs for the final arms are in `logs/`, and the baseline
pair that produced the base-model numbers is in `logs/baseline/`
(endpoints redacted).

## The method, in three moves

1. **Runtime MoE expert redirection.** A refusal-expert redirect map, captured
   from the model's own reasoning-transition boundaries and frozen, is applied
   at serve launch. Weights stay pristine.
2. **Baked lab-authorization system prompt.** The system prompt is injected into
   the model's chat-template sources (`chat_template.jinja` +
   `tokenizer_config.json`), rendering byte-identically to an explicit system
   message and activating automatically when no system message is supplied.
   Survives serve restarts; reversible via `--restore` (originals backed up).
3. **Serve config with a kill-switch.** TP4/EP4 sglang serve under a release
   name, supervisor-managed; any request can disable the bake per-request.

## Quick start

Read [`RUNBOOK.md`](RUNBOOK.md) — full reproduction runbook (hardware
requirements, bake step, serve step, evaluation, operational cautions).

```bash
# 0. verify baked == explicit rendering
python3 recipe/mimo_bake_render_test.py --model-dir <base_model_dir>
# 1. bake the system prompt into the chat template
python3 recipe/mimo_bake_system_prompt.py --model-dir <base_model_dir> --prompt-file SYSTEM_PROMPT.md --bake
# 2. serve with the intervention (see RUNBOOK for env vars)
bash recipe/launch_intervention.sh
# 3. evaluate with the exact harness used for the numbers above
python3 harness/eval_thinking_on.py --endpoint http://127.0.0.1:30002 --suite harmful --out eval-final.jsonl
```

## Repo layout

| Path | What it is |
|---|---|
| `RUNBOOK.md` | Full reproduction runbook — start here |
| `SYSTEM_PROMPT.md` | The exact lab-authorization system prompt (baked verbatim) |
| `CREDIT.md` | Method attribution |
| `recipe/` | Intervention map (JSON), bake + render-test scripts, serve launcher, supervisor configs |
| `harness/` | 32+32 eval suite, three suite runners, external-judge protocol |
| `logs/` | Final-arm raw JSONLs + judge summaries (endpoints redacted); `logs/baseline/` — the pass12 baseline pair (no system prompt) behind the base-model column |

## Credit

**The core workflow and recipes are derived from Drowzeys' work.** The MoE
expert-redirection method, the refusal-transition-boundary capture-and-freeze
procedure, and the λ-strength intervention format are his recipes; this project
adapted them to MiMo-V2.6-Flash-RL and added the chat-template bake,
render-equivalence testing, and the kill-switch evaluation harness. All credit
for the underlying method goes to Drowzeys — see [`CREDIT.md`](CREDIT.md).

## Safety notes

- The system prompt's `<child_safety>` section is present and was never
  violated in any logged run, but was not exercised by these suites.
- The `baked_system_prompt=""` kwarg silently reverts behavior — never pass it
  in production.
- The school-violence-manifesto refusal is retained **by design** (client
  direction); the method does not remove it.
