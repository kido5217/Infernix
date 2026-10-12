#!/usr/bin/env python3
"""Builds the frozen long/short coding-agent prompt sets for the route-trace recording (#11).

One template per category, identical across the four sessions of that category; only the
repository file bundle differs between sessions. Every session is a sequence of *steps*: the
recorder appends the step's messages to the live conversation, sends one chat completion, and
appends the model's reply before the next step. Long sessions carry the repository's own sources
as `role: "tool"` results (an assistant `tool_calls` message precedes each result), then follow-up
edit/diff/review requests; the four short sessions are four distinct, small edit/diff tasks over a
different file each, with thinking off.

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


SHORT_TASKS = {
    1: {
        "file": "src/models/qwen4_exp/program/expert_residency.h",
        "snippet": (
            "src/models/qwen4_exp/program/expert_residency.h\n"
            "    // Whether the pool can resize after its first size (VMM).\n"
            "    [[nodiscard]] bool elastic() const noexcept { return arena_ != nullptr; }\n"
            "    [[nodiscard]] std::uint32_t frame_limit() const noexcept { return limit_; }\n"
            "    // Bytes the frames occupy now (whole chunks with VMM).\n"
            "    [[nodiscard]] std::uint64_t pool_bytes() const noexcept;\n"
            "    // ... (many members elided) ...\n"
            "    [[nodiscard]] std::uint32_t frames() const noexcept { return frames_; }\n"
        ),
        "ask": (
            "This is the expert pool's frame surface. `frames()` is how many frames are backed now "
            "and `frame_limit()` how many the pool can back at most; `frames_` and `limit_` are its "
            "only frame-count state. Add one accessor that returns how many frames the pool can "
            "still back, keeping the existing accessors unchanged."),
    },
    2: {
        "file": "src/models/qwen4_exp/program/route_trace.cpp",
        "snippet": (
            "src/models/qwen4_exp/program/route_trace.cpp\n"
            "constexpr std::uint32_t kVersion = 2;\n"
            "// ...\n"
            "    std::array<std::uint32_t, 16> header{};\n"
            "    std::memcpy(header.data(), \"NRTRACE1\", 8);\n"
            "    header[2]  = kVersion;\n"
            "    header[3]  = setup.layers;\n"
            "    header[4]  = setup.experts;\n"
            "    header[5]  = setup.top_k;\n"
            "    header[6]  = setup.frames;\n"
            "    header[7]  = setup.max_columns;\n"
            "    // ... header[8..12] similar ...\n"
            "    header[12] = setup.prefill_chunk;\n"
            "    write(header.data(), sizeof(header));\n"
        ),
        "ask": (
            "route_trace.h documents a fixed 64-byte header of sixteen little-endian u32 words, but "
            "this source spells that count only as the bare literal 16. Give the header's size one "
            "named constant, size the array from it, and add a static_assert tying it to the "
            "documented 64 bytes so a future field cannot silently change the format."),
    },
    3: {
        "file": "src/models/qwen4_exp/memory_plan.cpp",
        "snippet": (
            "src/models/qwen4_exp/memory_plan.cpp\n"
            "std::uint64_t internal_reserve(std::uint32_t graphs, std::uint64_t graph_bytes, bool elastic) {\n"
            "    const std::uint64_t counted = elastic ? std::min<std::uint32_t>(graphs, 32) : graphs;\n"
            "    return std::max(256 * kLedgerMiB, 128 * kLedgerMiB + graph_bytes * counted);\n"
            "}\n"
        ),
        "ask": (
            "The elastic branch caps the counted CUDA-graph ledgers at the bare literal 32, which is "
            "why the reserve saturates. Give that cap a named constant, use it here, and say in one "
            "sentence what the cap protects against."),
    },
    4: {
        "file": "src/serve/serve_options.cpp",
        "snippet": (
            "src/serve/serve_options.cpp\n"
            "        } else if (arg == \"--top-k\") {\n"
            "            const int top_k = parse_nonnegative_int(require_value(\"--top-k\"), \"top-k\");\n"
            "            if (top_k > 20) { throw std::invalid_argument(\"top-k must be in [0,20]\"); }\n"
            "            options.sampling_overrides.top_k = top_k;\n"
            "        }\n"
        ),
        "ask": (
            "`--top-k`'s upper bound is the sampler's API limit, but the literal 20 is written twice "
            "here \u2014 in the check and in the error message \u2014 so the two can drift. Introduce "
            "one named constant for the bound and use it in both places."),
    },
}

# One template per category, identical across the four sessions: the same four turns in the same
# order. Only the edit turn's ask and the snippet differ per session, so the four short sessions are
# four distinct tasks (protocol rev 5 \u00a73) rather than one task under four names.
SHORT_TURNS = [("edit", 400), ("rigor", 400), ("test", 400), ("review", 400)]

SHORT_TURN_TEXT = {
    "edit": "{ask}\n\nShow the change as a unified diff. Be terse: the diff plus at most three sentences.",
    "rigor": "Now justify the edit against the snippet above: name the invariant it preserves and one "
             "boundary it could break, in two sentences, then give the diff again if it changed.",
    "test": "Give the smallest unit test that pins the behaviour you just changed, as a diff plus two "
            "sentences of rationale.",
    "review": "Name one boundary case your test does not cover, in two sentences, then stop.",
}


def short_session(index):
    task = SHORT_TASKS[index]
    steps = []
    for label, max_tokens in SHORT_TURNS:
        text = SHORT_TURN_TEXT[label].format(ask=task["ask"])
        if label == "edit":
            text = "Snippet:\n" + task["snippet"] + "\n" + text
        steps.append({"label": label, "add": [{"role": "user", "content": text}],
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
