"""The Dataset 2 tasks with fixed item sets and string item ids: BIG-Bench Hard (src/bbh_d2.py), MATH-500
(src/math_d2.py) and HotpotQA (src/hotpot_d2.py). The scripts' `--task` branches look the module up here; GSM8K keeps its own code path."""
from __future__ import annotations

from types import ModuleType

from . import bbh_d2, hotpot_d2, math_d2

TASKS = ("gsm8k", bbh_d2.TASK, math_d2.TASK, hotpot_d2.TASK)
MODULES: dict[str, ModuleType] = {bbh_d2.TASK: bbh_d2, math_d2.TASK: math_d2, hotpot_d2.TASK: hotpot_d2}


def module(task: str) -> ModuleType | None:
    """The task's pipeline module, or None for GSM8K."""
    return MODULES.get(task)
