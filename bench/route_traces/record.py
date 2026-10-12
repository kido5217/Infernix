#!/usr/bin/env python3
"""Records one route trace per coding-agent session with the locally built serve (#11).

For each session in bench/route_traces/prompts/ it starts `infernix-serve` at the deployed
Qwen3.8-Flash-Next flags plus `--route-trace <trace>` and `--log-level debug`, drives the
session's steps over the OpenAI chat-completions API, stops the server with SIGTERM (so the
Program's orderly shutdown writes the per-layer hit/miss diagnostic), and records the outcome.

Outputs (all under out/route-traces/, which is git-ignored):
  <name>.bin    the NRTRACE route trace
  <name>.jsonl  the serve request log
  <name>.log    the serve stdout/stderr (holds the per-layer Debug line)
  results.json  per-session per-step token counts and trace hashes

    python3 bench/route_traces/record.py [--only NAME ...] [--dry]

Run inside the repository's dev harness (`nix develop .`) so libcuda.so.1 resolves.
"""
import argparse
import glob
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROMPTS = os.path.join(HERE, "prompts")
OUT = os.path.join(ROOT, "out", "route-traces")

EXE = os.environ.get("INFERNIX_SERVE", os.path.join(ROOT, "build", "apps", "infernix-serve"))
ARTIFACT = os.environ.get(
    "INFERNIX_QWEN4_ARTIFACT",
    "/home/kido/ai/models/infernix/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix")
NGRAM = os.environ.get("INFERNIX_QWEN4_NGRAM",
                       "/home/kido/ai/models/infernix/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram")
HOST = "127.0.0.1"
PORT = int(os.environ.get("INFERNIX_PORT", "8828"))
MODEL_ID = "Qwen3.8-Flash-Next"

# The unit's flags (nixos-configs/users/kido/llm.nix, infernix-qwen38-flash-next) exactly, plus
# --log-level debug for the per-layer diagnostic. --request-log-jsonl and --route-trace are added
# per session by the recorder.
BASE_FLAGS = [
    "--ngram-volume", NGRAM,
    "--host", HOST, "--port", str(PORT), "--model-id", MODEL_ID,
    "--kv-dtype", "nvfp4",
    "--max-context", "1048576", "--rope-yarn-factor", "4", "--prefill-chunk", "4096",
    "--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft",
    "--max-concurrency", "4",
    "--expert-ram-mib", "66000",
    "--vision", "--preserve-thinking",
    "--reasoning-loop", "conclude", "--default-thinking-budget", "32768",
    "--max-pending-requests", "64", "--pending-timeout-ms", "600000", "--media-cache-mib", "2048",
    "--log-level", "debug",
]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        try:
            sock.connect((HOST, PORT))
            return True
        except OSError:
            return False


def model_ready():
    try:
        with urllib.request.urlopen("http://%s:%d/v1/models" % (HOST, PORT), timeout=5) as resp:
            return len(json.loads(resp.read().decode("utf-8")).get("data", [])) > 0
    except Exception:
        return False


def start_serve(trace, request_log, serve_log):
    args = [EXE, ARTIFACT] + BASE_FLAGS + ["--route-trace", trace, "--request-log-jsonl", request_log]
    log("LAUNCH: " + " ".join(args))
    handle = open(serve_log, "wb")
    proc = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("serve exited early rc=%s; see %s" % (proc.returncode, serve_log))
            if port_open() and model_ready():
                log("serve ready (pid %s)" % proc.pid)
                return proc, handle
            time.sleep(3)
        raise RuntimeError("serve not ready within 900s; see %s" % serve_log)
    except Exception:
        if proc.poll() is None:
            proc.kill()
        handle.close()
        raise


def stop_serve(proc, handle):
    log("stopping serve (SIGTERM, pid %s) to run the orderly shutdown" % proc.pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        log("serve did not stop in 600s; killing")
        proc.kill()
        proc.wait(timeout=60)
    handle.close()


def chat(session, messages, step):
    body = {"model": MODEL_ID, "messages": messages, "max_tokens": step["max_tokens"],
            "temperature": 1.0, "top_p": 0.95, "stream": False}
    if step.get("enable_thinking") is not None:
        # The engine reads the per-request thinking toggle from chat_template_kwargs.
        body["chat_template_kwargs"] = {"enable_thinking": bool(step["enable_thinking"])}
    if session.get("tools"):
        body["tools"] = session["tools"]
        body["tool_choice"] = step.get("tool_choice", "auto")
    request = urllib.request.Request("http://%s:%d/v1/chat/completions" % (HOST, PORT),
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    entry = payload["choices"][0]
    return entry["message"], entry.get("finish_reason"), payload.get("usage", {}), time.time() - started


def run_session(session_path):
    session = json.load(open(session_path, encoding="utf-8"))
    name = session["name"]
    trace = os.path.join(OUT, name + ".bin")
    request_log = os.path.join(OUT, name + ".jsonl")
    serve_log = os.path.join(OUT, name + ".log")
    for path in (trace, request_log, serve_log):
        if os.path.exists(path):
            os.remove(path)
    proc, handle = start_serve(trace, request_log, serve_log)
    steps = []
    try:
        messages = [{"role": "system", "content": session["system"]}]
        for index, step in enumerate(session["steps"]):
            messages.extend(step["add"])
            message, finish, usage, wall = chat(session, messages, step)
            reply = message.get("content") or ""
            steps.append({"label": step["label"], "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens"),
                          "finish_reason": finish, "wall_seconds": round(wall, 2)})
            log("%s step %d/%d %-7s prompt=%s completion=%s finish=%s %.1fs" % (
                name, index + 1, len(session["steps"]), step["label"], usage.get("prompt_tokens"),
                usage.get("completion_tokens"), finish, wall))
            messages.append({"role": "assistant", "content": reply})
    finally:
        stop_serve(proc, handle)

    with open(serve_log, "rb") as fh:
        log_text = fh.read().decode("utf-8", "replace")
    per_layer = [line for line in log_text.splitlines() if "expert cache per layer" in line]
    engine_hits = [line for line in log_text.splitlines() if "expert cache:" in line and "since start" in line]
    result = {
        "name": name, "category": session["category"],
        "prompt_file": os.path.relpath(session_path, ROOT),
        "prompt_sha256": hashlib.sha256(open(session_path, "rb").read()).hexdigest(),
        "serve_command": " ".join([EXE, ARTIFACT] + BASE_FLAGS + ["--route-trace", trace,
                                                                  "--request-log-jsonl", request_log]),
        "trace_path": os.path.relpath(trace, ROOT),
        "prompt_tokens_total": sum(s["prompt_tokens"] or 0 for s in steps),
        "max_prompt_tokens": max((s["prompt_tokens"] or 0 for s in steps), default=0),
        "completion_tokens_usage": sum(s["completion_tokens"] or 0 for s in steps),
        "steps": steps,
        "engine_per_layer_line": per_layer[-1] if per_layer else None,
        "engine_cache_line": engine_hits[-1] if engine_hits else None,
    }
    if os.path.exists(trace):
        with open(trace, "rb") as fh:
            data = fh.read()
        result["trace_bytes"] = len(data)
        result["trace_sha256"] = hashlib.sha256(data).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(PROMPTS, "*.json")))
    if args.only:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in args.only]
    if args.dry:
        for path in paths:
            session = json.load(open(path, encoding="utf-8"))
            print(session["name"], session["category"],
                  "steps=", len(session["steps"]),
                  "first-chars=", sum(len(json.dumps(s["add"])) for s in session["steps"][:1]))
        print("serve command:", " ".join([EXE, ARTIFACT] + BASE_FLAGS))
        return 0
    results_path = os.path.join(OUT, "results.json")
    results = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            results[name] = run_session(path)
            with open(results_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1, sort_keys=True)
        except Exception as error:
            log("SESSION %s FAILED: %s" % (name, error))
            results[name] = {"name": name, "error": str(error)}
            with open(results_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1, sort_keys=True)
            if isinstance(error, KeyboardInterrupt):
                raise
    summary = {k: (v.get("completion_tokens_usage"), v.get("trace_sha256", "ERR")[:12]) for k, v in results.items()}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
