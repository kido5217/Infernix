#!/usr/bin/env python3
"""Assembles the committed trace-set manifest from a recording's results.json + validation.

Reads out/route-traces/results.json (written by record.py), recomputes each trace's generated-token
count from the NRTRACE rounds (the protocol floor: sum of `tokens` over Decode + Verify + Forced
rounds), folds in the `host_tier.py --validate` verdict if out/route-traces/validate.txt exists,
and writes bench/route_traces/manifest.json (committed; the trace binaries themselves stay under
out/route-traces/, which is git-ignored).

    python3 bench/route_traces/manifest.py
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "out", "route-traces")
sys.path.insert(0, os.path.join(ROOT, "tools", "expert_cache_replay"))
import host_tier as ht  # noqa: E402

FLOOR = {"long": 1024, "short": 256}


def validate_verdict(name):
    path = os.path.join(OUT, "validate.txt")
    if not os.path.exists(path):
        return None
    lines = []
    with open(path, encoding="utf-8") as handle:
        current = None
        for line in handle:
            if line.startswith("==="):
                current = line.strip()
            elif current and name in current:
                lines.append(line.rstrip())
    return lines or None


def main():
    results = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    decode_settings = next(
        (r.get("decode_settings") for r in (results[n] for n in sorted(results)) if "error" not in r), None)
    entries = []
    for name in sorted(results, key=lambda n: (results[n].get("category", "?"), n)):
        result = results[name]
        if "error" in result:
            entries.append({"name": name, "error": result["error"]})
            continue
        trace_path = os.path.join(ROOT, result["trace_path"])
        trace = ht.read_trace(trace_path)
        generated = sum(r.tokens for r in trace.rounds if r.kind in (ht.DECODE, ht.VERIFY, ht.FORCED))
        floor = FLOOR[result["category"]]
        entry = {
            "name": name,
            "category": result["category"],
            "prompt_file": result["prompt_file"],
            "prompt_sha256": result["prompt_sha256"],
            "serve_command": result["serve_command"],
            "trace_path": result["trace_path"],
            "trace_sha256": result["trace_sha256"],
            "trace_bytes": result["trace_bytes"],
            "decode_settings": result.get("decode_settings", decode_settings),
            "trace_version": trace.header["version"],
            "trace_header_frames": trace.header["frames"],
            "rounds": len(trace.rounds),
            "generated_tokens": generated,
            "generated_tokens_floor": floor,
            "qualifies": generated >= floor,
            "max_prompt_tokens": result["max_prompt_tokens"],
            "completion_tokens_usage": result["completion_tokens_usage"],
            "steps": result["steps"],
            "engine_cache_line": result.get("engine_cache_line"),
            "validate": validate_verdict(name),
        }
        entries.append(entry)
    manifest = {
        "protocol": "wayfinder #11; floors long>=1024 / short>=256 generated tokens "
                    "(Decode+Verify+Forced rounds); 4 long (>=32K-token prompt) + 4 short (<=2K-token, "
                    "3-5-turn); greedy decoding (protocol rev 5 \u00a73) and eight distinct sessions",
        "decode_settings": decode_settings,
        "artifact": "/home/kido/ai/models/infernix/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix",
        "ngram_volume": "/home/kido/ai/models/infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram",
        "sessions": entries,
    }
    path = os.path.join(HERE, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=False)
    for entry in entries:
        if "error" in entry:
            print("%-8s ERROR %s" % (entry["name"], entry["error"]))
        else:
            print("%-8s %-5s gen=%-6d floor=%-5d %s  %s" % (
                entry["name"], entry["category"], entry["generated_tokens"], entry["generated_tokens_floor"],
                "OK" if entry["qualifies"] else "BELOW-FLOOR", entry["trace_sha256"][:12]))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
