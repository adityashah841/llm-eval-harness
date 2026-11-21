"""Loads YAML datasets into EvalSample objects.

A dataset file is either a plain list of samples, or a mapping with an
optional `system_prompt` (applied to every sample in the file) and a
`samples` list. A file without its own `system_prompt` (e.g. one written by
scripts/generate_tests.py) inherits the first one defined by another file in
the same folder. Each sample needs `id`, `prompt` and `expected`; optional
fields are `domain`, `tags`, `difficulty`, `context` (the source text the
answer must be grounded in — used as the NLI premise instead of `expected`)
and `tests` (Python asserts — marks the sample as code to be executed).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


class DatasetError(ValueError):
    pass


@dataclass
class EvalSample:
    id: str
    prompt: str
    expected: str
    domain: str
    tags: List[str] = field(default_factory=list)
    difficulty: Optional[str] = "medium"
    system_prompt: str = ""
    context: Optional[str] = None
    tests: Optional[str] = None


def _parse_file(f: Path) -> tuple[str, List[EvalSample]]:
    with open(f, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    system_prompt = ""
    if isinstance(data, dict):
        system_prompt = str(data.get("system_prompt") or "").strip()
        data = data.get("samples")
    if data is None:
        return system_prompt, []
    if not isinstance(data, list):
        raise DatasetError(f"{f.name}: expected a list of samples")

    samples = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise DatasetError(f"{f.name}: sample #{i + 1} is not a mapping")
        missing = [k for k in ("id", "prompt", "expected") if not item.get(k)]
        if missing:
            raise DatasetError(f"{f.name}: sample #{i + 1} is missing {', '.join(missing)}")
        samples.append(
            EvalSample(
                id=str(item["id"]),
                prompt=str(item["prompt"]),
                expected=str(item["expected"]),
                domain=item.get("domain", "general"),
                tags=item.get("tags", []) or [],
                difficulty=item.get("difficulty", "medium"),
                system_prompt=str(item.get("system_prompt") or ""),
                context=item.get("context"),
                tests=item.get("tests"),
            )
        )
    for s in samples:
        s.system_prompt = s.system_prompt or system_prompt
    return system_prompt, samples


def load_dataset(path: str) -> List[EvalSample]:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset not found: {path}")
    files = sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml"))) if p.is_dir() else [p]

    samples: List[EvalSample] = []
    folder_prompt = ""
    for f in files:
        file_prompt, file_samples = _parse_file(f)
        folder_prompt = folder_prompt or file_prompt
        samples.extend(file_samples)
    for s in samples:
        s.system_prompt = s.system_prompt or folder_prompt

    seen = set()
    for s in samples:
        if s.id in seen:
            raise DatasetError(f"Duplicate sample id in {path}: {s.id}")
        seen.add(s.id)
    return samples
