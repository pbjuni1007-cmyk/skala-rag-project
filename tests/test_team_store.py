"""Persistence boundaries and interruption recovery, all inside tmp_path."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from agents.state import Limits
from agents.store import RunStore
from test_team_contracts import example
from test_team_supervisor import (
    CONTEXT, IDENTITY, RUN_ID, decision, make_supervisor, run,
)


def minimal_state(**updates):
    state = {
        "contract_version": "agent-contract-v1", "run_id": RUN_ID,
        "context": deepcopy(CONTEXT), "identity": IDENTITY,
        "status": "running", "next_action": "supervisor", "step_count": 0,
        "attempts": {}, "pending": None, "last_error": None,
        "feedback": {}, "evidence_sufficient": False, "summaries": {},
        "results": {}, "report": None, "evaluation": None, "publication": None,
    }
    state.update(updates)
    return state


def test_result_files_are_relative_hashed_and_immutable(tmp_path):
    store = RunStore(tmp_path)
    payload = example("research_ok")
    reference = store.put("research", 1, payload)
    path = tmp_path / reference["path"]

    assert not Path(reference["path"]).is_absolute()
    assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.get(reference) == payload
    original = path.read_bytes()
    with pytest.raises(ValueError):
        store.put("research", 1, {"different": "response for the same call"})
    assert path.read_bytes() == original


@pytest.mark.parametrize("relative", ["../outside.json", "artifacts/../../outside.json", "/tmp/outside.json", "."])
def test_read_refs_cannot_escape_the_run_folder(tmp_path, relative):
    store = RunStore(tmp_path)
    reference = {"path": relative, "sha256": "a" * 64}
    with pytest.raises(ValueError):
        store.verify(reference)


def test_symlink_cannot_turn_a_relative_ref_into_an_outside_read(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"outside": true}')
    (root / "link.json").symlink_to(outside)
    reference = {"path": "link.json", "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):
        RunStore(root).verify(reference)


def test_symlink_cannot_redirect_a_checkpoint_write(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("keep this file")
    (root / "snapshot.json").symlink_to(outside)
    with pytest.raises(ValueError):
        RunStore(root).checkpoint(minimal_state())
    assert outside.read_text() == "keep this file"


def test_run_lock_rejects_another_writer_and_is_released_after_exception(tmp_path):
    first, second = RunStore(tmp_path), RunStore(tmp_path)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        with first.locked():
            with pytest.raises(ValueError):
                with second.locked():
                    pytest.fail("A second writer acquired the same run")
            raise RuntimeError("simulated interruption")
    with second.locked():
        second.checkpoint(minimal_state())
    assert (tmp_path / "snapshot.json").exists()


def test_snapshot_limit_counts_utf8_bytes_and_preserves_the_last_confirmed_state(tmp_path):
    store = RunStore(tmp_path)
    store.checkpoint(minimal_state())
    confirmed = (tmp_path / "snapshot.json").read_bytes()
    oversized = minimal_state(feedback={"writer": ["가" * (64 * 1024 // 3)]})
    with pytest.raises(ValueError, match="64 KiB"):
        store.checkpoint(oversized)
    assert (tmp_path / "snapshot.json").read_bytes() == confirmed


def test_loading_an_oversized_snapshot_is_rejected_before_recovery(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(minimal_state(feedback={"writer": ["x" * (64 * 1024)]})))
    with pytest.raises(ValueError, match="64 KiB"):
        RunStore(tmp_path).load(IDENTITY)


def test_result_mutation_is_detected_before_the_snapshot_is_resumed(tmp_path):
    store = RunStore(tmp_path)
    reference = store.put("research", 1, example("research_ok"))
    store.checkpoint(minimal_state(results={"research": reference}))
    (tmp_path / reference["path"]).write_text('{"tampered": true}')
    with pytest.raises(ValueError):
        store.load(IDENTITY)


def test_confirmed_research_is_reused_after_a_new_supervisor_resumes(tmp_path):
    first_supervisor, first_nodes, _ = make_supervisor(tmp_path)
    paused = run(first_supervisor, pause_after=2)
    assert paused["status"] == "running"
    assert paused["pending"] is None
    assert [role for role, _ in first_nodes.calls] == ["research"]

    resumed_supervisor, resumed_nodes, _ = make_supervisor(tmp_path)
    state = run(resumed_supervisor, resume=True)
    assert state["status"] == "completed"
    assert not resumed_nodes.requests("research")
    assert state["attempts"]["research"] == 1
    first_id = first_nodes.requests("research")[0]["request_id"]
    assert resumed_nodes.requests("domain")[0]["research_context"]["request_id"] == first_id


@pytest.mark.parametrize("role", ["supervisor", "research", "writer", "evaluator", "publish"])
def test_pending_is_saved_before_every_external_call_and_uncertain_calls_are_never_replayed(tmp_path, role):
    observed = []

    def interrupt(request, payload=None):
        snapshot = json.loads((tmp_path / "snapshot.json").read_text())
        assert snapshot["pending"] is not None
        assert request["request_id"] in json.dumps(snapshot["pending"])
        observed.append(request["request_id"])
        raise KeyboardInterrupt("Simulated process loss after request dispatch")

    kwargs = {"decider": interrupt} if role == "supervisor" else {"overrides": {role: interrupt}}
    supervisor, _, _ = make_supervisor(tmp_path, **kwargs)
    with pytest.raises(KeyboardInterrupt):
        run(supervisor)
    assert len(observed) == 1

    resumed_supervisor, resumed_nodes, resumed_decider = make_supervisor(tmp_path)
    state = run(resumed_supervisor, resume=True)
    assert state["status"] == "needs_attention"
    assert state["last_error"]["code"] == "uncertain_request"
    assert not resumed_nodes.calls
    assert not resumed_decider.requests


@pytest.mark.parametrize("changed", ["identity", "context", "limits", "run_id"])
def test_resume_rejects_changed_execution_inputs_without_making_a_call(tmp_path, changed):
    supervisor, _, _ = make_supervisor(tmp_path)
    run(supervisor, pause_after=2)
    previous_snapshot = (tmp_path / "snapshot.json").read_bytes()
    limits = Limits(worker_attempts=3) if changed == "limits" else None
    resumed, nodes, policy = make_supervisor(tmp_path, limits=limits)
    context = deepcopy(CONTEXT)
    if changed == "context":
        context["scenario"] = "Changed scope"
    identity = "different-code-data-config" if changed == "identity" else IDENTITY
    run_id = "different-run" if changed == "run_id" else RUN_ID

    with pytest.raises(ValueError):
        resumed.run(run_id, context, identity, resume=True)
    assert not nodes.calls
    assert not policy.requests
    assert (tmp_path / "snapshot.json").read_bytes() == previous_snapshot


def test_resume_checks_current_result_hashes_before_any_call(tmp_path):
    supervisor, _, _ = make_supervisor(tmp_path)
    paused = run(supervisor, pause_after=2)
    result_path = tmp_path / paused["results"]["research"]["path"]
    result_path.write_text('{"changed": true}')
    resumed, nodes, policy = make_supervisor(tmp_path)
    with pytest.raises(ValueError):
        run(resumed, resume=True)
    assert not nodes.calls
    assert not policy.requests


def test_completed_run_is_reused_without_republishing(tmp_path):
    supervisor, _, _ = make_supervisor(tmp_path)
    completed = run(supervisor)
    assert completed["status"] == "completed"
    resumed, nodes, policy = make_supervisor(tmp_path)
    assert run(resumed, resume=True) == completed
    assert not nodes.calls
    assert not policy.requests


def test_completed_resume_rejects_changed_published_files(tmp_path):
    supervisor, _, _ = make_supervisor(tmp_path)
    completed = run(supervisor)
    publication = RunStore(tmp_path).get(completed["publication"])
    (tmp_path / publication["pdf_ref"]["path"]).write_bytes(b"changed after publication")
    resumed, nodes, _ = make_supervisor(tmp_path)
    with pytest.raises(ValueError):
        run(resumed, resume=True)
    assert not nodes.calls


def test_large_source_payload_is_external_and_does_not_accumulate_in_state(tmp_path):
    marker = "source-only-marker " * 6000

    def long_source(request, payload):
        payload["chunks"][0]["text"] += marker
        return payload

    supervisor, _, _ = make_supervisor(tmp_path, overrides={"research": long_source})
    state = run(supervisor, pause_after=2)
    serialized = json.dumps(state, ensure_ascii=False)
    assert len((tmp_path / "snapshot.json").read_bytes()) <= 64 * 1024
    assert marker not in serialized
    assert set(state["results"]["research"]) == {"path", "sha256"}
    stored = RunStore(tmp_path).get(state["results"]["research"])
    assert marker in stored["chunks"][0]["text"]


def test_long_decision_reason_is_preserved_in_local_logs_not_control_state(tmp_path):
    reason = "local decision reason marker " * 3000

    def verbose_decider(request):
        result = decision(request, "research")
        return result.model_copy(update={"reason": reason})

    supervisor, _, _ = make_supervisor(tmp_path, decider=verbose_decider)
    state = run(supervisor, pause_after=1)
    assert reason not in json.dumps(state, ensure_ascii=False)
    assert len((tmp_path / "snapshot.json").read_bytes()) <= 64 * 1024
    assert reason in (tmp_path / "events.jsonl").read_text()
