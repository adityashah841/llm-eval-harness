import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from llm_eval.runner.dataset_loader import DatasetError, load_dataset

from ..config import datasets_dir
from ..schemas import DatasetInfo

router = APIRouter(tags=["datasets"])

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def resolve_dataset(dataset: str) -> tuple[str, Path]:
    """Map a request's dataset ('legal_qa' or 'datasets/legal_qa') to a folder
    inside the datasets directory. Anything else — absolute paths, '..',
    files elsewhere on the server — is rejected, so the API can't be used to
    probe the filesystem.

    Returns (storage key, folder). The key is always 'datasets/<name>' so runs
    group together in /metrics/timeseries however the dataset was named.
    """
    name = dataset.strip().replace("\\", "/")
    if name.startswith("datasets/"):
        name = name[len("datasets/"):]
    name = name.rstrip("/")
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
    root = datasets_dir()
    folder = (root / name).resolve()
    if folder.parent != root or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
    return f"datasets/{name}", folder


@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """Datasets that can be passed to POST /runs."""
    root = datasets_dir()
    infos = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and _NAME_RE.match(p.name)):
        try:
            samples = load_dataset(str(folder))
        except DatasetError:
            continue
        if not samples:
            continue
        infos.append(
            DatasetInfo(
                name=folder.name,
                path=f"datasets/{folder.name}",
                sample_count=len(samples),
                scoring="unit_tests" if all(s.tests for s in samples) else "rouge_nli",
            )
        )
    return infos
