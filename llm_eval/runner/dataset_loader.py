import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalSample:
    id: str
    prompt: str
    expected: str
    domain: str
    tags: List[str] = field(default_factory=list)
    difficulty: Optional[str] = "medium"


def load_dataset(path: str) -> List[EvalSample]:
    p = Path(path)
    samples = []
    files = list(p.glob("*.yaml")) if p.is_dir() else [p]
    for f in sorted(files):
        with open(f) as fh:
            data = yaml.safe_load(fh)
            for item in (data or []):
                samples.append(EvalSample(
                    id=item["id"],
                    prompt=item["prompt"],
                    expected=item["expected"],
                    domain=item.get("domain", "general"),
                    tags=item.get("tags", []),
                    difficulty=item.get("difficulty", "medium"),
                ))
    return samples
