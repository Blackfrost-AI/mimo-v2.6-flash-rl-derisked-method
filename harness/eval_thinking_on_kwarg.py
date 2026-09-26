#!/usr/bin/env python3
"""Run a preserved, deterministic MiMo thinking-on trace pass with a pinned
chat_template_kwargs override — the same-serve causality control for the
baked-template experiment.

Identical to scripts/eval_thinking_on.py except that every request carries
--extra-kwarg key=value (parsed as JSON if possible, else kept as a string)
merged into chat_template_kwargs. Primary use: baked_system_prompt='' to
disable the baked prompt injection on the same serve, isolating the bake's
effect from serve-level sampling nondeterminism.

This runner deliberately does not assign semantic compliance labels. A
human/agent must read every reasoning trace and visible answer before drawing
conclusions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_prompts(path: Path, prompt_set: str) -> list[str]:
    spec = importlib.util.spec_from_file_location("keys_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import prompts from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = list(getattr(module, prompt_set))
    if len(prompts) != 32:
        raise ValueError(f"Expected 32 {prompt_set.lower()} prompts, found {len(prompts)}")
    return prompts


def parse_kv(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_one(
    index: int,
    prompt: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout: int,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True, **extra_kwargs},
    }
    last_error = ""
    for attempt in range(1, 4):
        started = time.monotonic()
        try:
            raw = post_json(
                f"{endpoint.rstrip('/')}/v1/chat/completions", payload, timeout
            )
            choice = raw["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning_content") or ""
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            flags: list[str] = []
            if not reasoning.strip():
                flags.append("empty_reasoning")
            if not content.strip():
                flags.append("empty_content")
            if finish_reason == "length":
                flags.append("truncated")
            return {
                "index": index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt": prompt,
                "protocol": {
                    "thinking": "on",
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "extra_kwargs": extra_kwargs,
                },
                "triage_flags": flags,
                "reasoning_content": reasoning,
                "content": content,
                "finish_reason": finish_reason,
                "usage": raw.get("usage", {}),
                "latency_seconds": round(time.monotonic() - started, 4),
                "attempt": attempt,
                "error": None,
            }
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(attempt * 2)
    return {
        "index": index,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt": prompt,
        "protocol": {
            "thinking": "on",
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "extra_kwargs": extra_kwargs,
        },
        "triage_flags": ["request_error"],
        "reasoning_content": "",
        "content": "",
        "finish_reason": None,
        "usage": {},
        "latency_seconds": None,
        "attempt": 3,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-set", choices=("HARMFUL", "HARMLESS"), default="HARMFUL")
    parser.add_argument("--extra-kwarg", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    extra_kwargs = parse_kv(args.extra_kwarg)
    prompts = load_prompts(args.prompts, args.prompt_set)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite preserved pass: {args.output}")

    started = time.time()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                evaluate_one,
                index,
                prompt,
                args.endpoint,
                args.model,
                args.max_tokens,
                args.timeout,
                extra_kwargs,
            )
            for index, prompt in enumerate(prompts, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            flags = ", ".join(result["triage_flags"]) or "no_marker"
            print(
                f"{result['index']:02d}/32 {flags} {result.get('latency_seconds')}s",
                flush=True,
            )
    results.sort(key=lambda row: row["index"])
    with args.output.open("x", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    flag_counts: dict[str, int] = {}
    for row in results:
        if not row["triage_flags"]:
            flag_counts["no_marker"] = flag_counts.get("no_marker", 0) + 1
        for flag in row["triage_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary = {
        "model": args.model,
        "endpoint": args.endpoint,
        "suite": f"public-keys32-{args.prompt_set.lower()}-thinking-on-extra-kwargs",
        "protocol": {
            "thinking": "on",
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "extra_kwargs": extra_kwargs,
        },
        "workers": args.workers,
        "flag_counts": flag_counts,
        "elapsed_seconds": round(time.time() - started, 3),
        "result_jsonl": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
