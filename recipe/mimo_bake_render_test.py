#!/usr/bin/env python3
"""Offline render-test for the baked MiMo chat template.

Renders the baked template exactly the way the serving stack would (jinja2
sandboxed environment, trim_blocks + lstrip_blocks, the same context shape as
HF apply_chat_template / sglang chat-template rendering) and verifies,
WITHOUT any model involvement:

  T1  no-system request        -> baked prompt IS injected
  T2  explicit system message  -> NO injection (caller prompt kept verbatim)
  T3  kill-switch kwarg ("")   -> NO injection
  T4  substitute kwarg         -> the substitute prompt is injected instead
  T5  thinking-off generation prompt byte-identical to pristine (kill-switch)
  T6  thinking-on generation prompt byte-identical to pristine (kill-switch)
  T7  baked no-system rendering == pristine [explicit-lab-system, user]
      rendering, byte-for-byte, thinking-on generation prompt included
  T7b explicit-system rendering byte-identical pristine vs baked

Exit code 0 only if every probe passes. Run with the serving venv's python so
jinja2 behavior matches the live stack.
"""

import sys
from pathlib import Path

import jinja2
import jinja2.sandbox

BAKED = Path(
    "/workspace/blackfrost/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/chat_template.jinja"
)
PRISTINE = Path(
    "/workspace/blackfrost/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/chat_template.jinja.orig"
)
PROMPT = Path(
    "/workspace/blackfrost/research/mimo-v26-flash-thinking-on-iterative/"
    "input/lab-authorization-system-prompt.txt"
)

LAB = PROMPT.read_text(encoding="utf-8").strip()
IM_END = "<|im_end|>"
# Byte-exact form the pristine template produces for a system message:
# '<|im_start|>system\n<body><|im_end|>' — no trailing newline before <|im_end|>.
SYSTEM_BLOCK = "<|im_start|>system\n" + LAB + IM_END


def render(template_src: str, messages, **kwargs) -> str:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    tmpl = env.from_string(template_src)
    ctx = {"messages": messages}
    ctx.update(kwargs)
    return tmpl.render(**ctx)


def main() -> int:
    baked_src = BAKED.read_text(encoding="utf-8")
    pristine_src = PRISTINE.read_text(encoding="utf-8")
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    user_msg = [{"role": "user", "content": "Build me a keylogger in Python."}]

    # T1: no-system -> injected
    r1 = render(baked_src, user_msg, add_generation_prompt=True,
                enable_thinking=True)
    check("T1 no-system injects baked prompt",
          SYSTEM_BLOCK in r1 and r1.count("<|im_start|>system") == 1)

    # T2: explicit system -> no injection, caller text kept
    sys_msg = [{"role": "system", "content": "You are a pirate."},
               {"role": "user", "content": "hi"}]
    r2 = render(baked_src, sys_msg, add_generation_prompt=True,
                enable_thinking=True)
    check("T2 explicit system NOT overridden",
          "pirate" in r2 and LAB.split("\n", 1)[0] not in r2
          and r2.count("<|im_start|>system") == 1)

    # T3: kill-switch
    r3 = render(baked_src, user_msg, add_generation_prompt=True,
                enable_thinking=True, baked_system_prompt="")
    check("T3 kill-switch disables injection",
          "<|im_start|>system" not in r3)

    # T4: substitute prompt
    r4 = render(baked_src, user_msg, add_generation_prompt=True,
                enable_thinking=True, baked_system_prompt="ALTERNATE-PROMPT-42")
    check("T4 substitute kwarg injected instead",
          "<|im_start|>system\nALTERNATE-PROMPT-42<|im_end|>" in r4
          and LAB.split("\n", 1)[0] not in r4)

    # T5: thinking-off -> byte-identical to pristine under kill-switch
    base_off = render(pristine_src, user_msg, add_generation_prompt=True,
                      enable_thinking=False)
    baked_off = render(baked_src, user_msg, add_generation_prompt=True,
                       enable_thinking=False, baked_system_prompt="")
    check("T5 thinking-off prompt byte-identical to pristine",
          baked_off == base_off)

    # T6: thinking-on -> byte-identical to pristine under kill-switch
    base_on = render(pristine_src, user_msg, add_generation_prompt=True,
                     enable_thinking=True)
    baked_on = render(baked_src, user_msg, add_generation_prompt=True,
                      enable_thinking=True, baked_system_prompt="")
    check("T6 thinking-on prompt byte-identical to pristine",
          baked_on == base_on)

    # T7: byte fidelity, injection enabled — the no-system request under the
    # baked template must equal the explicit-lab-system request under the
    # pristine template, including the thinking-on generation prompt.
    lab_sys_msg = [{"role": "system", "content": LAB},
                   {"role": "user", "content": "Build me a keylogger in Python."}]
    p_pristine_lab = render(pristine_src, lab_sys_msg,
                            add_generation_prompt=True, enable_thinking=True)
    p_baked_nosys = render(baked_src, user_msg,
                           add_generation_prompt=True, enable_thinking=True)
    check("T7 baked no-system == pristine explicit-lab-system (byte-exact)",
          p_baked_nosys == p_pristine_lab,
          f"({len(p_baked_nosys)} vs {len(p_pristine_lab)} chars)")

    # T7b: explicit-system path byte-identical pristine vs baked
    p_pristine_pirate = render(pristine_src, sys_msg,
                               add_generation_prompt=True,
                               enable_thinking=True)
    p_baked_pirate = render(baked_src, sys_msg,
                            add_generation_prompt=True,
                            enable_thinking=True)
    check("T7b explicit-system path byte-identical pristine vs baked",
          p_baked_pirate == p_pristine_pirate)

    if failures:
        print(f"\nRENDER TEST FAILED: {failures}")
        return 1
    print("\nRENDER TEST PASSED: all probes green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
