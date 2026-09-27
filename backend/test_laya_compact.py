"""Contract tests for laya_compact.py with a fake agent (no torch, no weights).

    uv run --with pytest pytest -q backend
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import laya_compact  # noqa: E402

QUESTION = {"type": "choice", "instructions": "q", "criteria": {"keep": "a", "truncate": "b", "drop": "c"}}


class FakeAgent:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((states, questions, kwargs))
        print("library chatter on stdout")  # must not reach the answer channel
        return [{"answers": {"q": {"probabilities": {"keep": 0.1 * (i + 1), "truncate": 0.2, "drop": 0.3}}}}
                for i, _ in enumerate(states)]


@pytest.fixture
def agent(monkeypatch):
    fake = FakeAgent()
    monkeypatch.setattr(laya_compact, "load_agent", lambda model, device, offline: fake)
    return fake


def run(monkeypatch, capfd, stdin, argv=()):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = laya_compact.main(list(argv))
    out, err = capfd.readouterr()
    return code, out, err


def request(**overrides):
    body = {"model": "typed-decisions", "max_len": 512, "head_max_len": 128, "batch_size": 32,
            "question": QUESTION, "states": [{"id": "t1", "state": {"a": 1}}, {"id": "t2", "state": "s"}]}
    body.update(overrides)
    return json.dumps(body)


def test_scores_every_state_and_prints_one_json_line(monkeypatch, capfd, agent):
    code, out, err = run(monkeypatch, capfd, request())
    assert code == 0
    lines = out.strip().splitlines()
    assert len(lines) == 1
    answer = json.loads(lines[0])
    assert answer["model"] == "typed-decisions"
    assert answer["device"] == "cpu"
    assert set(answer["scores"]) == {"t1", "t2"}
    assert answer["scores"]["t2"]["keep"] == pytest.approx(0.2)
    assert "library chatter" in err
    states, questions, kwargs = agent.calls[0]
    assert states == [{"a": 1}, "s"]
    assert questions == {"q": QUESTION}
    assert kwargs == {"batch_size": 32, "max_len": 512, "head_max_len": 128, "sort_by_length": True}
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_empty_states_skip_inference(monkeypatch, capfd, agent):
    code, out, _ = run(monkeypatch, capfd, request(states=[]))
    assert code == 0
    assert json.loads(out)["scores"] == {}
    assert agent.calls == []


@pytest.mark.parametrize("body", [
    "not json",
    "[]",
    request(model="jev-latest"),
    request(model="relative/path"),
    request(question={"type": "noul", "instructions": "x"}),
    request(states=[{"state": "no id"}]),
    request(states=[{"id": "t1", "state": 1}, {"id": "t1", "state": 2}]),
])
def test_bad_input_exits_2(monkeypatch, capfd, agent, body):
    code, out, err = run(monkeypatch, capfd, body)
    assert code == laya_compact.EXIT_BAD_INPUT
    assert out == ""
    assert "laya_compact:" in err


def test_accepts_a_fine_tuned_checkpoint_path(monkeypatch, capfd, agent):
    code, out, _ = run(monkeypatch, capfd, request(model="/abs/finetuned"))
    assert code == 0
    assert json.loads(out)["model"] == "/abs/finetuned"


def test_rejects_a_directory_that_is_not_a_checkpoint(monkeypatch, capfd, tmp_path):
    code, out, err = run(monkeypatch, capfd, request(model=str(tmp_path)))
    assert code == laya_compact.EXIT_BAD_INPUT
    assert out == ""
    assert "not a Laya checkpoint directory" in err


def test_missing_weights_exit_3(monkeypatch, capfd):
    def not_cached(model, offline):
        raise laya_compact.Failure(laya_compact.EXIT_NOT_CACHED, "weights are not cached")

    monkeypatch.setattr(laya_compact, "checkpoint_dir", not_cached)
    code, out, err = run(monkeypatch, capfd, request())
    assert code == laya_compact.EXIT_NOT_CACHED
    assert out == ""
    assert "not cached" in err


def test_unexpected_errors_exit_1(monkeypatch, capfd):
    def boom(model, device, offline):
        raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(laya_compact, "load_agent", boom)
    code, out, err = run(monkeypatch, capfd, request())
    assert code == laya_compact.EXIT_ERROR
    assert out == ""
    assert "RuntimeError: MPS out of memory" in err
