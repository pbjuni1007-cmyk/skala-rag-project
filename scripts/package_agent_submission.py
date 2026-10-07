"""Verify one completed Agent run and package its reviewed submission files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfReader

from agents.store import RunStore


def _label(value):
    if not value or not value.strip() or any(char in value for char in "/\\\r\n\0") or value in {".", ".."}:
        raise ValueError("Invalid submission filename field")
    return value


def package_submission(*, run_dir, traces, git_url, campus, class_name, contributors,
                       output_dir, real_run_confirmed=False, visual_review_confirmed=False):
    """Package verified files; human confirmations are required for non-code checks."""
    if not real_run_confirmed or not visual_review_confirmed:
        raise ValueError("Confirm real execution and visual PDF review before packaging")
    if not re.fullmatch(r"[0-9]+반", class_name):
        raise ValueError("Class name must use the X반 format")
    campus = _label(campus)
    contributors = "+".join(_label(name) for name in contributors.split("+"))
    parsed = urlparse(git_url)
    if (parsed.scheme != "https" or parsed.netloc != "github.com" or
            not parsed.path.startswith("/pbjuni1007-cmyk/skala-rag-project/tree/") or
            not parsed.path.removeprefix("/pbjuni1007-cmyk/skala-rag-project/tree/")):
        raise ValueError("Provide the team GitHub branch link")

    run_dir = Path(run_dir).resolve()
    if not (run_dir / "snapshot.json").is_file():
        raise ValueError("Completed run snapshot is missing")
    store = RunStore(run_dir)
    state = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    if state.get("status") != "completed" or state.get("run_id") != run_dir.name:
        raise ValueError("Submission requires one completed run directory")
    if not state.get("publication") or not state.get("report") or not state.get("evaluation"):
        raise ValueError("Report, evaluation and publication must all exist")
    result = store.get(state["publication"])
    report = store.get(state["report"])
    evaluation = store.get(state["evaluation"])
    if (result.get("status") != "ok" or result.get("run_id") != state["run_id"] or
            report.get("run_id") != state["run_id"] or
            evaluation.get("run_id") != state["run_id"] or
            evaluation.get("report_request_id") != report.get("request_id") or
            evaluation.get("passed") is not True or result.get("human_review_pending") is not True or
            set(evaluation.get("checks", {})) != {"groundedness", "neutrality", "bias_control", "coverage"} or
            any(check.get("passed") is not True for check in evaluation["checks"].values())):
        raise ValueError("Published report and evaluation do not match this run")
    markdown = store.verify(result["markdown_ref"])
    pdf = store.verify(result["pdf_ref"])
    if markdown.decode("utf-8") != report["markdown"]:
        raise ValueError("Published Markdown differs from the evaluated report")
    pdf_path = run_dir / result["pdf_ref"]["path"]
    reader = PdfReader(pdf_path)
    pages = len(reader.pages)
    if (not 1 <= pages == result.get("pdf_pages") <= 10 or
            "SUMMARY" not in reader.pages[0].extract_text() or
            "REFERENCE" not in reader.pages[-1].extract_text()):
        raise ValueError("Final PDF fails page or chapter checks")

    files = [Path(path) for path in traces]
    files.sort(key=lambda path: int(match.group(1)) if (match := re.fullmatch(
        r"tracing-([0-9]+)\.png", path.name)) else -1)
    if not files or [path.name for path in files] != [f"tracing-{index}.png"
                                                     for index in range(1, len(files) + 1)]:
        raise ValueError("Trace PNG files must be numbered from tracing-1.png")
    for path in files:
        if not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Trace evidence must be PNG files")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"Agent_{campus}_{class_name}_{contributors}.zip"
    if destination.exists():
        raise ValueError("Submission ZIP already exists")
    manifest = {
        "run_id": state["run_id"], "git_branch_url": git_url,
        "pdf_pages": pages, "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
        "trace_files": [path.name for path in files],
        "real_run_confirmed": True, "visual_review_confirmed": True,
    }
    with NamedTemporaryFile(dir=output_dir, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("report.md", markdown)
            archive.writestr(pdf_path.name, pdf)
            for path in files:
                archive.write(path, path.name)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--git-url", required=True)
    parser.add_argument("--campus", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-real-run", action="store_true")
    parser.add_argument("--confirm-visual-review", action="store_true")
    args = parser.parse_args()
    print(package_submission(
        run_dir=args.run_dir, traces=args.trace, git_url=args.git_url,
        campus=args.campus, class_name=args.class_name, contributors=args.contributors,
        output_dir=args.output_dir, real_run_confirmed=args.confirm_real_run,
        visual_review_confirmed=args.confirm_visual_review,
    ))


if __name__ == "__main__":
    main()
