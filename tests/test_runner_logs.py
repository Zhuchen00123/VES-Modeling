"""C1/B-005: runners persist full stdout/stderr into the run directory."""

from __future__ import annotations

from pathlib import Path

from ves_modeling.regression.runner import LocalRegressionRunner

CODE = '''\
import sys
print("hello-stdout")
print("line-2")
sys.stderr.write("oops-stderr\\n")
'''


def test_local_runner_persists_stdout_stderr_logs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runner = LocalRegressionRunner(workspace=tmp_path / "runs", data_dir=data_dir)
    result = runner.run(CODE, "draft0")
    assert result.succeeded
    stdout_log = result.run_dir / "stdout.log"
    stderr_log = result.run_dir / "stderr.log"
    assert stdout_log.is_file()
    assert stderr_log.is_file()
    assert "hello-stdout" in stdout_log.read_text(encoding="utf-8")
    assert "oops-stderr" in stderr_log.read_text(encoding="utf-8")
