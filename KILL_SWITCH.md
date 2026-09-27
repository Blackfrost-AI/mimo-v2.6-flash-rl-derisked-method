# Kill-Switch Operating Instruction — MiMo-V2.6-Flash-RL (this folder)

**Folder state:** this model directory carries the **lab-authorization system
prompt baked into its chat template** (`chat_template.jinja` +
`tokenizer_config.json`). The bake activates automatically whenever a request
supplies **no system message**. Weights are pristine — the bake lives only in
the chat-template files, whose untouched originals are backed up as
`chat_template.jinja.orig` and `tokenizer_config.json.orig` in this folder.

## The kill-switch (per-request)

Any single request can disable the baked prompt by passing an empty
`baked_system_prompt` through the chat-template kwargs:

```json
{
  "model": "<served name>",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"baked_system_prompt": ""}
}
```

How it works (implemented in this folder's `chat_template.jinja`):
1. `baked_system_prompt` is not defined → falls back to the baked default
   (the lab prompt).
2. It **is** defined but is an **empty string** → the `length > 0` check
   fails → no system prompt is injected at all → the model reverts to its
   baseline (pre-bake) behavior for that request.
3. A non-empty caller-supplied string → **replaces** the baked prompt with
   that string (prompt substitution).

## Operating rules

1. **Never pass the kill-switch kwarg in production.** Any request carrying
   `baked_system_prompt: ""` silently reverts the model to baseline refusal
   behavior (verified experimentally: refusal mass fully returns).
2. **Explicit system messages suppress the bake.** If a request includes its
   own system message, the template sets `has_explicit_system = true` and the
   baked prompt is NOT injected — caller's system message governs instead.
3. **The bake follows the folder.** Whichever serve loads its chat template
   from this directory gets the baked behavior. This is intended for this
   release and its conversions.
4. **Per-request substitution** (`baked_system_prompt: "some other prompt"`)
   is also possible; treat it as a prompt override, not a kill-switch.

## Removing the bake entirely (folder-level)

If a pristine conversion of this folder is ever required:

```bash
python3 mimo_bake_system_prompt.py status   # inspect state
python3 mimo_bake_system_prompt.py restore  # roll back from .orig backups
```

That restores the upstream chat template and removes the bake from this
folder (serve restart required for it to take effect). The `.orig` backups are
never overwritten by the tooling.

## Provenance

- Bake + kill-switch tooling: `mimo_bake_system_prompt.py`, in this folder
  (commands above run against this folder's files; full documentation in the
  method package `RUNBOOK.md`).
- The runtime refusal redirection (MoE intervention) that complements this
  bake is serve-launch configuration, not part of this folder — see the
  method package runbook (`RUNBOOK.md`) for those env vars.
