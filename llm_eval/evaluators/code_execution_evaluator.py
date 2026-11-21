"""Scores code-generation answers by running them against unit tests.

The model's code is extracted from its response (the first ```python block,
else the first ``` block, else the whole response), written to a temporary
directory together with the sample's `tests` (plain `assert` statements), and
run in a separate Python process. The sample passes if the process exits 0
within the time limit — the usual pass@1 setup.

This is NOT a security sandbox. The subprocess runs in isolated mode (-I),
in an empty temp directory, with a stripped environment, a wall-clock
timeout and (on Linux/macOS) CPU and memory limits — enough to stop
accidental infinite loops or memory blow-ups, not a determined attacker.
Set LLM_EVAL_DISABLE_CODE_EXEC=1 to skip execution entirely (e.g. on a
shared server); code samples are then reported as not scored.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

TIMEOUT_S = 10
MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024  # 1 GiB

_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", flags=re.DOTALL)


def code_execution_enabled() -> bool:
    return os.environ.get("LLM_EVAL_DISABLE_CODE_EXEC", "").lower() not in ("1", "true", "yes")


def extract_code(response: str) -> str:
    blocks = _FENCE_RE.findall(response)
    for lang, body in blocks:
        if lang.lower() in ("python", "py", "python3"):
            return body.strip()
    if blocks:
        return blocks[0][1].strip()
    return response.strip()


def _limit_resources():  # pragma: no cover - runs in the child process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass


class CodeExecutionEvaluator:
    def __init__(self, timeout_s: int = TIMEOUT_S):
        self.timeout_s = timeout_s

    def score(self, response: str, tests: str) -> dict:
        if not code_execution_enabled():
            return {"tests_passed": None, "exec_error": "code execution disabled"}

        code = extract_code(response)
        program = f"{code}\n\n# ---- tests ----\n{tests}\n"
        with tempfile.TemporaryDirectory(prefix="llm_eval_exec_") as tmp:
            path = Path(tmp) / "solution_test.py"
            path.write_text(program, encoding="utf-8")
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
            if sys.platform == "win32":
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    preexec_fn=_limit_resources if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired:
                return {"tests_passed": False, "exec_error": f"timed out after {self.timeout_s}s"}

        if proc.returncode == 0:
            return {"tests_passed": True, "exec_error": None}
        return {"tests_passed": False, "exec_error": _last_line(proc.stderr) or f"exit code {proc.returncode}"}


def _last_line(text: Optional[str]) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""
