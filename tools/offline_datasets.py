"""Offline question files shared by the OM benchmark entry points (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def add_arguments(parser):
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset-dir", type=Path,
                        help="read *.jsonl/*.json question files directly inside this offline directory")
    inputs.add_argument("--dataset-files", type=Path, nargs="+", metavar="FILE",
                        help="read only these offline question files")
    parser.add_argument("--num-questions", type=int,
                        help="first N questions from EACH file; omitted: all questions")
    parser.add_argument("--dataset-field", default="question",
                        help="text field in each record (default: question; e.g. prompt)")
    parser.add_argument("--include-builtin-prompts", action="store_true",
                        help="combine selected short/long prompts with offline dataset questions in one run")


def selected(args):
    return bool(getattr(args, "dataset_dir", None) or getattr(args, "dataset_files", None))


def load(args):
    """Return frozen prompts and file provenance. Never download or truncate text."""
    directory, files = getattr(args, "dataset_dir", None), getattr(args, "dataset_files", None)
    if directory and files:
        raise ValueError("--dataset-dir and --dataset-files are mutually exclusive")
    if not getattr(args, "include_builtin_prompts", False) and (getattr(args, "prompts", None) or getattr(args, "prompt_id", None)
            or getattr(args, "prompt_group", "all") != "all"):
        raise ValueError("offline datasets cannot be combined with --prompts/--prompt-id/--prompt-group filtering without --include-builtin-prompts")
    limit = getattr(args, "num_questions", None)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("--num-questions must be positive (per file); omit it to read all")
    field = getattr(args, "dataset_field", "question")
    if not isinstance(field, str) or not field.strip():
        raise ValueError("--dataset-field must be a nonempty field name")
    if directory:
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"dataset directory not found: {directory}")
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in (".jsonl", ".json"))
    if not files:
        raise ValueError("no offline .jsonl/.json dataset files selected")
    paths = [Path(p).expanduser().resolve() for p in files]
    if len(set(paths)) != len(paths) or len({p.name for p in paths}) != len(paths):
        raise ValueError("dataset files must have distinct paths and filenames; rename duplicate filenames")
    prompts, datasets, ids = [], [], set()
    for path in paths:
        if path.suffix not in (".jsonl", ".json"):
            raise ValueError(f"unsupported dataset format: {path}; use .jsonl or a .json array")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_-")[:32] or "dataset"
        dataset_id = slug + "_" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:8]
        if dataset_id in ids:
            raise ValueError(f"dataset ID collision: {path.name}; rename the file")
        ids.add(dataset_id)
        digest, rows, count = hashlib.sha256(), [], 0

        def append(value, line):
            nonlocal count
            text = value if isinstance(value, str) else value.get(field) if isinstance(value, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line}: expected nonempty string or object with string field {field!r}")
            count += 1
            if limit is None or count <= limit:
                rows.append({"id": f"{dataset_id}_{count:06d}", "prompt": text,
                             "category": path.stem, "group": "custom", "dataset_id": dataset_id,
                             "dataset_file": path.name, "dataset_line": line})

        with path.open("rb") as stream:
            if path.suffix == ".jsonl":
                for line, raw in enumerate(stream, 1):
                    digest.update(raw)
                    try:
                        text = raw.decode("utf-8-sig" if line == 1 else "utf-8")
                        if text.strip():
                            append(json.loads(text), line)
                    except (UnicodeError, json.JSONDecodeError) as error:
                        raise ValueError(f"{path}:{line}: invalid UTF-8/JSON: {error}") from error
            else:
                raw = stream.read()
                digest.update(raw)
                try:
                    values = json.loads(raw.decode("utf-8-sig"))
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}: invalid UTF-8/JSON: {error}") from error
                if not isinstance(values, list):
                    raise ValueError(f"{path}: JSON dataset must be an array")
                for index, value in enumerate(values, 1):
                    append(value, index)
        if not count:
            raise ValueError(f"empty dataset file: {path}")
        datasets.append({"id": dataset_id, "name": path.name, "path": str(path),
                         "sha256": digest.hexdigest(), "field": field, "format": path.suffix[1:],
                         "total_samples": count, "selected_samples": len(rows),
                         "num_questions": limit, "selection": "first N records" if limit else "all records",
                         "line_unit": "line (1-based)" if path.suffix == ".jsonl" else "array item (1-based)"})
        prompts.extend(rows)
    return prompts, datasets


def prompt_label(prompt):
    if prompt.get("dataset_file"):
        return f"{prompt['dataset_file']}:{prompt['dataset_line']} ({prompt['id']})"
    return prompt["id"]
