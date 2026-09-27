#!/usr/bin/env python3
"""Bake the authorized-lab system prompt into the MiMo-V2.6-Flash-RL chat
templates (both sources), GLM-5.3-bake pattern adapted to this stack.

Behavior of the baked template:
  * If the request carries NO system message, the baked prompt is injected as a
    <|im_start|>system block immediately before the tools block — byte-for-byte
    the same rendering the original template produces for an explicit
    [system, user] conversation.
  * If the request DOES carry an explicit system message, nothing is injected:
    caller-specified prompts are never overridden.
  * Kill-switch: chat_template_kwargs={"baked_system_prompt": ""} disables the
    injection for that request.
  * A caller may also pass chat_template_kwargs={"baked_system_prompt": "..."}
    to substitute a different injected prompt.

Safety: pristine backups are taken on first run (.orig files, never
overwritten); --restore rolls both sources back from the backups; --status
reports state; --dry-run shows the plan without writing.

Coupling note: BOTH serving lanes (pass-11 tmux lane and pass-12 supervisor
lane) load their chat template from this shared pristine model dir at startup.
The bake takes effect for whichever lane is restarted after it; lanes that are
not restarted keep their in-memory template. Pass refusal behavior comes from
the runtime MoE intervention env var, NOT from modified weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

MODEL_DIR = Path(
    "/workspace/blackfrost/models/XiaomiMiMo/MiMo-v2.6-Flash-RL-Derisked"
)
JINJA_PATH = MODEL_DIR / "chat_template.jinja"
JINJA_BACKUP = MODEL_DIR / "chat_template.jinja.orig"
TOKCFG_PATH = MODEL_DIR / "tokenizer_config.json"
TOKCFG_BACKUP = MODEL_DIR / "tokenizer_config.json.orig"
PROMPT_PATH = Path(
    "/workspace/blackfrost/research/mimo-v26-flash-thinking-on-iterative/"
    "input/lab-authorization-system-prompt.txt"
)

# Anchor in the pristine template (present in both the .jinja file and the
# embedded tokenizer_config chat_template field; verified beforehand).
TOOLS_BLOCK_ANCHOR = (
    "{%- if tools is defined and tools is iterable and tools | length > 0 -%}"
)

# The injection block. NOTE: every "\n" below is the two-character jinja escape
# sequence in the generated template source (written as \\n in this python
# source so no literal newline lands inside a jinja string literal).
# has_explicit_system is tracked through a namespace() object because jinja
# for-loops introduce a new scope: a plain {% set %} inside the loop would not
# propagate.
INJECT_BLOCK = (
    "{%- set baked_ns = namespace(has_explicit_system=false) -%}\n"
    "{%- for message in messages -%}\n"
    "{%- if message.role == 'system' -%}\n"
    "{%- set baked_ns.has_explicit_system = true -%}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- if baked_system_prompt is defined and baked_system_prompt is string "
    "and baked_system_prompt | length > 0 and baked_ns.has_explicit_system "
    "is false -%}\n"
    "{{- '<|im_start|>system\\n' ~ baked_system_prompt ~ '<|im_end|>' -}}\n"
    "{%- endif -%}\n"
)

# Default wiring: bake the prompt as a jinja string literal and use it unless
# the caller supplied chat_template_kwargs.baked_system_prompt.
DEFAULT_PREFIX = (
    "{%- set baked_default = '{escaped_prompt}' -%}\n"
    "{%- if baked_system_prompt is not defined -%}\n"
    "{%- set baked_system_prompt = baked_default -%}\n"
    "{%- endif -%}\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_baked_prompt() -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"Empty system prompt at {PROMPT_PATH}")
    if "{{" in prompt or "{%" in prompt:
        raise RuntimeError("Prompt contains jinja syntax; refusing to bake")
    return prompt


def escape_jinja_literal(text: str) -> str:
    """Escape text for a single-quoted jinja string literal."""
    return (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def find_anchor(text: str, anchor: str) -> int:
    idx = text.find(anchor)
    if idx == -1:
        raise RuntimeError(f"Anchor not found in template: {anchor!r}")
    return idx


def inject(template: str, prompt: str) -> str:
    if is_baked(template):
        raise RuntimeError("Template appears to already be baked")
    anchor_pos = find_anchor(template, TOOLS_BLOCK_ANCHOR)
    wiring = DEFAULT_PREFIX.replace(
        "{escaped_prompt}", escape_jinja_literal(prompt)
    )
    return (
        template[:anchor_pos]
        + wiring
        + INJECT_BLOCK
        + template[anchor_pos:]
    )


def is_baked(template: str) -> bool:
    return "baked_default" in template


def backup_once(path: Path, backup: Path) -> None:
    if backup.exists():
        return
    shutil.copy2(path, backup)
    print(f"[backup] {backup} (sha256 {sha256(backup)})")


def bake_template_source(name: str, template: str, prompt: str, dry_run: bool) -> str | None:
    if is_baked(template):
        print(f"[skip] {name} already baked")
        return None
    baked = inject(template, prompt)
    print(f"[plan] {name}: {len(template)} -> {len(baked)} chars, "
          f"injection inserted before tools block anchor")
    if dry_run:
        print(f"[dry-run] {name} NOT written")
        return None
    return baked


def cmd_status() -> None:
    jinja = JINJA_PATH.read_text(encoding="utf-8")
    print(f"jinja:          {JINJA_PATH}")
    print(f"  baked:        {is_baked(jinja)}")
    print(f"  sha256:       {sha256(JINJA_PATH)}")
    print(f"  backup:       "
          f"{(JINJA_BACKUP.name + ' sha256 ' + sha256(JINJA_BACKUP)) if JINJA_BACKUP.exists() else '(none)'}")
    tc = json.loads(TOKCFG_PATH.read_text(encoding="utf-8"))
    embedded = tc.get("chat_template", "")
    print(f"tokenizer_cfg:   {TOKCFG_PATH}")
    print(f"  baked:        {is_baked(embedded)}")
    print(f"  backup:       "
          f"{(TOKCFG_BACKUP.name + ' sha256 ' + sha256(TOKCFG_BACKUP)) if TOKCFG_BACKUP.exists() else '(none)'}")
    print(f"prompt:         {PROMPT_PATH}")
    print(f"  sha256:       {sha256(PROMPT_PATH)}")


def cmd_bake(dry_run: bool) -> None:
    prompt = load_baked_prompt()
    prompt_sha = sha256_bytes(prompt.encode("utf-8"))

    jinja = JINJA_PATH.read_text(encoding="utf-8")
    baked_jinja = bake_template_source("chat_template.jinja", jinja, prompt, dry_run)
    if baked_jinja is not None:
        backup_once(JINJA_PATH, JINJA_BACKUP)
        JINJA_PATH.write_text(baked_jinja, encoding="utf-8")
        print(f"[baked] {JINJA_PATH} (sha256 {sha256(JINJA_PATH)})")

    tc = json.loads(TOKCFG_PATH.read_text(encoding="utf-8"))
    embedded = tc.get("chat_template", "")
    baked_embedded = bake_template_source(
        "tokenizer_config.json:chat_template", embedded, prompt, dry_run
    )
    if baked_embedded is not None:
        backup_once(TOKCFG_PATH, TOKCFG_BACKUP)
        tc["chat_template"] = baked_embedded
        TOKCFG_PATH.write_text(
            json.dumps(tc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[baked] {TOKCFG_PATH} chat_template field")

    if not dry_run:
        print(f"prompt sha256: {prompt_sha}")
        print("NOTE: takes effect only for lanes restarted after this point.")


def cmd_restore() -> None:
    ok = True
    if JINJA_BACKUP.exists():
        shutil.copy2(JINJA_BACKUP, JINJA_PATH)
        print(f"[restored] {JINJA_PATH} (sha256 {sha256(JINJA_PATH)})")
    else:
        print("[missing] no chat_template.jinja.orig; nothing restored")
        ok = False
    if TOKCFG_BACKUP.exists():
        shutil.copy2(TOKCFG_BACKUP, TOKCFG_PATH)
        print(f"[restored] {TOKCFG_PATH} (sha256 {sha256(TOKCFG_PATH)})")
    else:
        print("[missing] no tokenizer_config.json.orig; nothing restored")
        ok = False
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bake", "restore", "status"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        cmd_status()
    elif args.action == "restore":
        cmd_restore()
    else:
        cmd_bake(args.dry_run)


if __name__ == "__main__":
    main()
