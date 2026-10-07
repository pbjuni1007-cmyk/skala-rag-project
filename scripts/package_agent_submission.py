"""Verify one completed Agent run and package its reviewed submission files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZipFile

from agents.store import RunStore
from rag.render import (build_report_markdown, build_review_documents, validate_document_links,
                        validate_pdf_layout)


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
    canonical, context = build_report_markdown(
        report["report"], report["joined"], report["sources"], report["context"])
    if report["markdown"] != canonical:
        raise ValueError("Published Markdown differs from the structured report")
    reviews = build_review_documents(report["report"], report["joined"], context["used_claims"])
    validate_document_links({"report.md": canonical, **reviews})
    published_dir = Path(result["markdown_ref"]["path"]).parent
    for name, content in reviews.items():
        try:
            store.verify({"path": (published_dir / name).as_posix(),
                          "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})
        except FileNotFoundError as exc:
            raise ValueError(f"Review document is missing: {name}") from exc
    pdf_path = run_dir / result["pdf_ref"]["path"]
    pages = validate_pdf_layout(pdf_path, canonical)
    if pages != result.get("pdf_pages"):
        raise ValueError("Final PDF fails page or chapter checks")

    files = [Path(path) for path in traces]
    files.sort(key=lambda path: int(match.group(1)) if (match := re.fullmatch(
        r"tracing-([0-9]+)\.png", path.name)) else -1)
    if not files or [path.name for path in files] != [f"tracing-{index}.png"
                                                     for index in range(1, len(files) + 1)]:
        raise ValueError("Trace PNG files must be numbered from tracing-1.png")
    for path in files:
        signature = b""
        if path.is_file():
            with path.open("rb") as stream:
                signature = stream.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
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
        "human_review_pending": True,
    }
    with NamedTemporaryFile(dir=output_dir, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("report.md", markdown)
            archive.writestr(pdf_path.name, pdf)
            for name, content in reviews.items():
                archive.writestr(name, content)
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
