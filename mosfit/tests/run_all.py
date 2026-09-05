"""Run every MOSFiT package test script.

Discovers ``mosfit/tests/_test_*.py`` and the repo-root generative smoke
``test.py``. This is what GitHub Actions CI invokes.
"""
from __future__ import print_function

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent


def main():
    scripts = sorted(TESTS.glob('_test_*.py'))
    root_smoke = ROOT / 'test.py'
    jobs = list(scripts)
    if root_smoke.is_file():
        jobs.append(root_smoke)
    if not jobs:
        print('no test scripts found', file=sys.stderr)
        return 1

    failed = []
    env = os.environ.copy()
    env.setdefault('MOSFIT_PHOTOMETRY_DEBUG', '')
    for script in jobs:
        rel = script.relative_to(ROOT)
        print('========', rel, '========', flush=True)
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            env=env,
        )
        if proc.returncode != 0:
            failed.append(str(rel))
            print('FAILED', rel, 'exit', proc.returncode, flush=True)

    if failed:
        print('failed:', ', '.join(failed), file=sys.stderr)
        return 1
    print('all package tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
