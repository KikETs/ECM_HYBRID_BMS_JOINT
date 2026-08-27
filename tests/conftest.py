import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')


def run(args, cwd=ROOT):
    """Run a repo script and return (returncode, stdout, stderr)."""
    p = subprocess.run([sys.executable] + args, cwd=cwd,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


@pytest.fixture
def sandbox_tables(tmp_path, monkeypatch):
    """A throwaway copy of results/tables that tests may corrupt.

    Nothing here ever writes to the real tables directory.
    """
    dst = tmp_path / 'tables'
    shutil.copytree(TABLES, dst)
    return dst
