"""Single-writer local run storage; no credentials or automatic remote I/O."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from agents.contracts import ArtifactRef, VIEWS


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root / ".run.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another process is already executing this run") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _path(self, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Artifact path must be relative to this run")
        resolved = (self.root / path).resolve()
        if resolved == self.root or not resolved.is_relative_to(self.root):
            raise ValueError("Artifact path escapes this run")
        return resolved

    def _atomic(self, path, data, *, replace=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            if replace:
                temporary.replace(path)
            else:
                # link is exclusive: a previous or ambiguous result is never replaced.
                os.link(temporary, path)
        except FileExistsError:
            raise ValueError("Refusing to overwrite an existing result") from None
        finally:
            temporary.unlink(missing_ok=True)

    def checkpoint(self, state):
        data = encoded(state)
        if len(data) > 64 * 1024:
            raise ValueError("Control state exceeds 64 KiB; keep payloads in artifacts")
        self._atomic(self._path("snapshot.json"), data)

    def put(self, role, attempt, payload):
        if role not in {"supervisor", *VIEWS, "writer", "evaluator", "publish"}:
            raise ValueError("Unknown artifact role")
        if type(attempt) is not int or attempt < 1:
            raise ValueError("Attempt must be a positive integer")
        relative = f"artifacts/{role}/{attempt}.json"
        data = encoded(payload)
        self._atomic(self._path(relative), data, replace=False)
        return {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}

    def verify(self, ref):
        value = ArtifactRef.model_validate(ref)
        data = self._path(value.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != value.sha256:
            raise ValueError("Artifact content changed")
        return data

    def get(self, ref):
        value = ArtifactRef.model_validate(ref)
        if not Path(value.path).parts or Path(value.path).parts[0] != "artifacts":
            raise ValueError("Result JSON must be in the artifact directory")
        return json.loads(self.verify(value))

    def load(self, identity):
        with self._path("snapshot.json").open("rb") as stream:
            data = stream.read(64 * 1024 + 1)
        if len(data) > 64 * 1024:
            raise ValueError("Control state exceeds 64 KiB")
        state = json.loads(data)
        if state["identity"] != identity:
            raise ValueError("Resume rejected: code, data, context or configuration changed")
        refs = list(state.get("results", {}).values())
        refs += [state.get(key) for key in ("report", "evaluation", "publication", "repair_source")]
        for ref in refs:
            if ref is not None:
                self.get(ref)
        if state.get("publication"):
            result = self.get(state["publication"])
            if result["status"] == "ok":
                self.verify(result["markdown_ref"])
                self.verify(result["pdf_ref"])
        return state

    def event(self, **event):
        with self._path("events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
