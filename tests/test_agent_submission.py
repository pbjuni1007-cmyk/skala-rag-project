"""Mechanical ZIP checks use fictional run and PNG data, not submission evidence."""
import base64
import json
from zipfile import ZipFile

import pytest

from scripts.package_agent_submission import package_submission
from test_agent_runtime_integration import RUN_ID, execute


GIT_URL = "https://github.com/pbjuni1007-cmyk/skala-rag-project/tree/work/%234-hong"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
    "lQAAAABJRU5ErkJggg=="
)


def inputs(tmp_path):
    state, _, _, root = execute(tmp_path)
    assert state["status"] == "completed"
    trace = tmp_path / "tracing-1.png"
    trace.write_bytes(PNG)
    return {
        "run_dir": root, "traces": [trace], "git_url": GIT_URL,
        "campus": "판교", "class_name": "7반",
        "contributors": "김기현+김도현+박병준+홍수정", "output_dir": tmp_path / "submissions",
        "real_run_confirmed": True, "visual_review_confirmed": True,
    }


def test_package_checks_same_run_pdf_hash_pages_and_names(tmp_path):
    options = inputs(tmp_path)
    archive = package_submission(**options)
    assert archive.name == "Agent_판교_7반_김기현+김도현+박병준+홍수정.zip"
    with ZipFile(archive) as package:
        names = package.namelist()
        assert "submission.json" in names and "report.md" in names
        assert "tracing-1.png" in names and any(name.endswith(".pdf") for name in names)
        manifest = json.loads(package.read("submission.json"))
        assert manifest["run_id"] == RUN_ID and 1 <= manifest["pdf_pages"] <= 10
        assert manifest["git_branch_url"] == GIT_URL


def test_package_requires_human_review_confirmations(tmp_path):
    options = inputs(tmp_path)
    options["visual_review_confirmed"] = False
    with pytest.raises(ValueError, match="Confirm real execution"):
        package_submission(**options)
    assert not options["output_dir"].exists()


def test_package_rejects_tampered_pdf(tmp_path):
    options = inputs(tmp_path)
    root = options["run_dir"]
    state = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    result = json.loads((root / state["publication"]["path"]).read_text(encoding="utf-8"))
    (root / result["pdf_ref"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Artifact content changed"):
        package_submission(**options)


@pytest.mark.parametrize("name", ["trace.png", "tracing-2.png"])
def test_package_requires_numbered_trace_files(tmp_path, name):
    options = inputs(tmp_path)
    wrong = tmp_path / name
    wrong.write_bytes(PNG)
    options["traces"] = [wrong]
    with pytest.raises(ValueError, match="numbered"):
        package_submission(**options)
