#!/usr/bin/env python3
"""Builds the frozen long/short coding-agent prompt sets for the route-trace recording (#11).

One template per category, identical across the four sessions of that category; only the
repository file bundle differs between sessions. Every session is a sequence of *steps*: the
recorder appends the step's messages to the live conversation, sends one chat completion, and
appends the model's reply before the next step. Long sessions carry the repository's own sources
as `role: "tool"` results (an assistant `tool_calls` message precedes each result), then follow-up
edit/diff/review requests; short sessions are a terse edit/diff exchange over a small snippet.

Deterministic: the file slices are fixed by (path, start, length) and the wording is literal, so
re-running regenerates byte-identical prompt files.

    python3 bench/route_traces/make_prompts.py            # writes prompts/*.json

Run it from anywhere; paths resolve from this file's location.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROMPT_DIR = os.path.join(HERE, "prompts")

LONG_SYSTEM = (
    "You are an expert software-engineering agent working in the Infernix repository, a "
    "from-scratch C++/CUDA inference engine for single-GPU LLM serving. You read the code before "
    "editing it, make the smallest correct change, and verify with the project's build and tests.\n\n"
    "Ground rules:\n"
    "- Never guess file contents; the tool results below are the ground truth.\n"
    "- Prefer minimal, idiomatic diffs that match the surrounding code.\n"
    "- State the invariant a change preserves and any boundary it touches.\n"
    "- Lead with the outcome, then the supporting detail.\n"
)

SHORT_SYSTEM = (
    "You are a precise code reviewer. Given a small snippet you make one minimal, correct edit and "
    "show it as a unified diff. Be terse: the diff plus at most three sentences.\n"
)

# A small, realistic function-calling surface (kept tiny on purpose: the trace cares about the
# routed tokens, not the tool schema).
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read the contents of a file at a repository path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep_search", "description": "Search file contents for a regular expression.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"},
                                                        "path": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact text span within a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "old_string": {"type": "string"},
                                                        "new_string": {"type": "string"}},
                       "required": ["old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the repository test suite (or a named target).",
        "parameters": {"type": "object", "properties": {"target": {"type": "string"}},
                       "required": ["target"]}}},
]


def slice_file(path, start, length):
    """A deterministic slice of a repository file, marked as an excerpt when truncated."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        text = handle.read()
    if start or length < len(text):
        body = text[start:start + length]
        mark = f"  # [excerpt: {path} chars {start}..{start + len(body)} of {len(text)}]"
        return body + mark
    return text


# (session suffix, [(path, start, length), ...]). Each bundle is sized so the first request's
# prompt clears the protocol's >= 32K-token long-session bound with margin.
LONG_BUNDLES = {
    1: [("docs/maintainer/engine-architecture.md", 0, 10 ** 9),
        ("src/models/qwen4_exp/program/expert_residency.h", 0, 10 ** 9),
        ("src/models/qwen4_exp/program/expert_residency.cpp", 0, 10 ** 9),
        ("docs/maintainer/resource-scheduling-and-context-cache.md", 0, 10 ** 9),
        ("src/models/qwen4_exp/memory_plan.cpp", 0, 10 ** 9),
        ("src/models/qwen4_exp/program/program_impl.h", 0, 60000)],
    2: [("docs/maintainer/hybrid-prefix-cache-spec.md", 0, 10 ** 9),
        ("src/models/qwen4_exp/program/route_trace.h", 0, 10 ** 9),
        ("src/models/qwen4_exp/program/host_expert_tier.h", 0, 10 ** 9)],
    3: [("docs/maintainer/qwen3_8-flash-next-design.md", 0, 140000),
        ("src/models/qwen4_exp/program/host_expert_tier.h", 0, 10 ** 9)],
    4: [("src/models/qwen4_exp/program/program_impl.h", 0, 170000),
        ("src/serve/serve_options.cpp", 0, 10 ** 9)],
}

LONG_STEPS = [
    ("read", 3072,
     "I am reviewing the routed-expert cache and its supporting types. The tool results below are "
     "the exact contents of the files I have open. Read them and give me, for each file, the "
     "invariants it enforces and the boundary it owns. Be concrete about flags, counters and the "
     "lifetime rules."),
    ("edit", 3072,
     "Good. Now make one minimal change: give the expert residency a per-layer view of its routed "
     "and hit counters, exposed without allocating on the decode path, and state the exact diff as "
     "a unified diff against the header you were shown. Do not change any numerical behavior."),
    ("review", 3072,
     "Review your own diff adversarially. Name the boundary cases it could break, say which of the "
     "files above each risk lives in, and describe the smallest test that would catch a regression "
     "in the per-layer counters."),
    ("test", 3072,
     "Finally, write the test itself, in the style of the repository's existing tests. Show the "
     "full test body and the exact CTest command that runs it."),
]


def long_session(index):
    steps = []
    tools_used = []
    for (path, start, length) in LONG_BUNDLES[index]:
        tools_used.append((path, slice_file(path, start, length)))
    for label, max_tokens, prompt in LONG_STEPS:
        add = []
        if label == "read":
            add.append({"role": "user", "content": prompt})
            calls = []
            for n, (path, _) in enumerate(tools_used):
                calls.append({"id": f"call_l{index}_{n}", "type": "function",
                              "function": {"name": "read_file", "arguments": json.dumps({"path": path})}})
            add.append({"role": "assistant",
                        "content": "I will read the open files before answering.",
                        "tool_calls": calls})
            for n, (path, body) in enumerate(tools_used):
                add.append({"role": "tool", "tool_call_id": f"call_l{index}_{n}",
                            "content": f"$ read_file({path})\n{body}"})
        else:
            add.append({"role": "user", "content": prompt})
        steps.append({"label": label, "add": add, "max_tokens": max_tokens,
                      "enable_thinking": True})
    return {"name": f"long-{index}", "category": "long", "system": LONG_SYSTEM,
            "tools": TOOLS, "steps": steps}


SHORT_SNIPPET = (
    "src/models/qwen4_exp/program/expert_residency.h\n"
    "    [[nodiscard]] std::uint32_t frames() const noexcept { return frames_; }\n"
    "    [[nodiscard]] std::uint32_t frame_limit() const noexcept { return limit_; }\n"
    "    [[nodiscard]] bool elastic() const noexcept { return arena_ != nullptr; }\n"
)

SHORT_STEPS = [
    ("fix", 400,
     "Snippet:\n" + SHORT_SNIPPET +
     "This is the expert pool's frame-count surface. Add one accessor that returns the bytes the "
     "frames occupy now, keeping the four existing accessors unchanged. Show a unified diff and at "
     "most three sentences of rationale."),
    ("rigor", 400,
     "Now make `frame_limit()` constexpr-correct and explain in two sentences why the change is or "
     "is not a behavior change."),
    ("test", 400,
     "Give the smallest unit test that pins `frames()` and the new accessor on an empty pool, as a "
     "diff plus two sentences."),
    ("review", 400,
     "Name one boundary case your test does not cover, in two sentences, then stop."),
]


def short_session(index):
    steps = []
    for label, max_tokens, prompt in SHORT_STEPS:
        steps.append({"label": label, "add": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "enable_thinking": False})
    return {"name": f"short-{index}", "category": "short", "system": SHORT_SYSTEM,
            "tools": [], "steps": steps}


def main():
    os.makedirs(PROMPT_DIR, exist_ok=True)
    written = []
    for index in range(1, 5):
        for session in (long_session(index), short_session(index)):
            path = os.path.join(PROMPT_DIR, session["name"] + ".json")
            text = json.dumps(session, indent=1, ensure_ascii=False) + "\n"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            written.append((session["name"], session["category"], len(text), digest))
    for name, category, size, digest in written:
        print(f"{name:8s} {category:5s} {size:8d} bytes  sha256={digest}")


if __name__ == "__main__":
    main()
