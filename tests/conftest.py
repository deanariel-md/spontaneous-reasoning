"""Pytest configuration: make the repository root importable as `src.*`, and skip a test that needs a file of the
study's outputs (data/, results/, paper/numbers.tex, manifests/human_audit/) when that file is not unpacked here."""
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEPOSIT = re.compile(r"(?:^|[\\/])(data|results|docs[\\/]rating_two_raters|paper[\\/]numbers\.tex|paper[\\/]tables)"
                      r"(?:[\\/]|$)")
_MISSING = ("No such file", "is absent", "cannot read", "does not exist", "not found")


def _deposit_file(exc):
    """The missing deposit path behind an exception (or its causes), or None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        names = [str(getattr(exc, "filename", None) or "")] + re.findall(r"[^\s'\"]+", str(exc))
        missing = isinstance(exc, FileNotFoundError) or any(m in str(exc) for m in _MISSING)
        for name in names:
            if not name or not missing:
                continue
            p = Path(name)
            rel = p.relative_to(ROOT) if p.is_absolute() and ROOT in p.parents else p
            if _DEPOSIT.search(str(rel)) and not (ROOT / rel).exists():
                return str(rel).replace("\\", "/")
        exc = exc.__cause__ or exc.__context__
    return None


def _skip_if_deposit_missing(outcome):
    exc = outcome.excinfo[1] if outcome.excinfo else None
    if exc is None or isinstance(exc, pytest.skip.Exception):
        return
    rel = _deposit_file(exc)
    if rel:
        outcome.force_exception(pytest.skip.Exception(f"needs {rel} from the study's outputs", _use_item_location=True))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    outcome = yield
    _skip_if_deposit_missing(outcome)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    _skip_if_deposit_missing(outcome)
