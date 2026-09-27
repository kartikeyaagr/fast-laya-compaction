#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["laya==0.3.20"]
# ///
"""Scores fast-laya-compaction's per-call states with a local Laya checkpoint.

One shot, spawned by the plugin for every compaction (no daemon): it reads a
request on stdin, answers one `choice` question over every state with
`Agent.predict_batch`, and prints exactly one JSON line on stdout. Anything
else (library logs, warnings, progress bars) goes to stderr.

    stdin:  {"model" (a checkpoint name, or the absolute path of a fine-tuned
             checkpoint directory), "device"?, "max_len", "head_max_len", "batch_size",
             "question": {"type": "choice", ...}, "states": [{"id", "state"}]}
    stdout: {"model", "device", "load_ms", "infer_ms", "scores": {id: {label: p}}}

Exit codes: 0 ok, 2 bad input, 3 checkpoint not cached, 1 anything else.
The hook runs this offline; `--warmup` is the one-time online setup that
downloads the checkpoint and runs a dummy prediction.
"""
import json
import os
import sys
import time

# Checkpoint name -> subfolder of the bundle repo, as laya/router.py maps them.
REPO = "convaiinnovations/laya"
# The reviewed commit Laya itself pins (laya/revisions.py, newer than 0.3.20).
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
CHECKPOINTS = {"typed-decisions": "typed-decisions", "multilingual": "multilingual", "english": None}

EXIT_ERROR, EXIT_BAD_INPUT, EXIT_NOT_CACHED = 1, 2, 3

WARMUP_REQUEST = {
    "model": "typed-decisions",
    "max_len": 512,
    "head_max_len": 128,
    "batch_size": 32,
    "question": {
        "type": "choice",
        "instructions": "What should happen to this tool call?",
        "criteria": {"keep": "still needed", "drop": "obsolete"},
    },
    "states": [
        {"id": "w1", "state": {"tool_call": "Read file_path=src/a.ts", "superseded": True}},
        {"id": "w2", "state": {"tool_call": "Bash command=npm test", "outcome": "error, 812 chars"}},
    ],
}


class Failure(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def parse_request(text: str) -> dict:
    try:
        request = json.loads(text)
    except ValueError as error:
        raise Failure(EXIT_BAD_INPUT, "request is not JSON: %s" % error)
    if not isinstance(request, dict):
        raise Failure(EXIT_BAD_INPUT, "request must be a JSON object")
    model = request.get("model")
    if model not in CHECKPOINTS and not (isinstance(model, str) and os.path.isabs(model)):
        raise Failure(EXIT_BAD_INPUT, "model must be one of %s, or an absolute checkpoint path"
                      % ", ".join(CHECKPOINTS))
    question = request.get("question")
    if not isinstance(question, dict) or question.get("type") != "choice":
        raise Failure(EXIT_BAD_INPUT, "question must be a choice question")
    states = request.get("states")
    if not isinstance(states, list) or not all(
        isinstance(s, dict) and isinstance(s.get("id"), str) and "state" in s for s in states
    ):
        raise Failure(EXIT_BAD_INPUT, "states must be a list of {id, state}")
    if len({s["id"] for s in states}) != len(states):
        raise Failure(EXIT_BAD_INPUT, "state ids must be unique")
    return request


def checkpoint_dir(model: str, offline: bool) -> str:
    """The pinned snapshot holding one checkpoint, downloaded unless offline."""
    from huggingface_hub import snapshot_download

    subfolder = CHECKPOINTS[model]
    prefix = "%s/" % subfolder if subfolder else ""
    try:
        path = snapshot_download(REPO, revision=REVISION, allow_patterns=[
            prefix + name for name in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
        ])
    except Exception:
        if not offline:
            raise
        path = ""
    if not os.path.isfile(os.path.join(path, prefix + "model.safetensors")):
        raise Failure(EXIT_NOT_CACHED, "Laya %s weights are not cached; run: "
                      "uv run --script backend/laya_compact.py --warmup" % model)
    return path


def load_agent(model: str, device, offline: bool):
    local = model not in CHECKPOINTS
    if local and not os.path.isfile(os.path.join(model, "rl_agent_config.json")):
        raise Failure(EXIT_BAD_INPUT, "%s is not a Laya checkpoint directory" % model)
    path = model if local else checkpoint_dir(model, offline)
    import torch
    import laya

    cpu = device == "cpu" or (not device and not torch.backends.mps.is_available()
                              and not torch.cuda.is_available())
    if cpu:
        # Laya's CPU benchmarks: default inter-op threading is ~10x slower.
        torch.set_num_interop_threads(1)
    return laya.load(path, subfolder=None if local else CHECKPOINTS[model], device=device or None)


def score(request: dict, offline: bool) -> dict:
    started = time.perf_counter()
    agent = load_agent(request["model"], request.get("device") or None, offline)
    loaded = time.perf_counter()
    states = request["states"]
    results = agent.predict_batch(
        [s["state"] for s in states],
        {"q": request["question"]},
        batch_size=request.get("batch_size"),
        max_len=request.get("max_len"),
        head_max_len=request.get("head_max_len"),
        sort_by_length=True,
    ) if states else []
    finished = time.perf_counter()
    return {
        "model": request["model"],
        "device": str(agent.device),
        "load_ms": round((loaded - started) * 1000),
        "infer_ms": round((finished - loaded) * 1000),
        "scores": {s["id"]: r["answers"]["q"]["probabilities"] for s, r in zip(states, results)},
    }


def main(argv) -> int:
    warmup = "--warmup" in argv
    if not warmup:
        # The hook must never download: a missing checkpoint fails fast (exit 3).
        os.environ["HF_HUB_OFFLINE"] = "1"
    # Keep stdout for the answer alone: whatever the libraries print, from
    # Python or native code, lands on stderr.
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        request = WARMUP_REQUEST if warmup else parse_request(sys.stdin.read())
        answer = score(request, offline=not warmup)
    except Failure as failure:
        print("laya_compact: %s" % failure, file=sys.stderr)
        return failure.code
    except Exception as error:  # noqa: BLE001 - the hook only needs a code and a message
        print("laya_compact: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return EXIT_ERROR
    if warmup:
        print("laya_compact: ready (%s on %s, load %d ms, infer %d ms)"
              % (answer["model"], answer["device"], answer["load_ms"], answer["infer_ms"]),
              file=sys.stderr)
    out.write(json.dumps(answer, separators=(",", ":")) + "\n")
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
