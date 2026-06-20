# Offline replay-eval harness (phase 3, Workstream 0).
#
# Everything phase 3 judges goes through this package; no new-model number is
# trusted until reproduce_phase2.py passes its gate on the 90-game phase-2
# set. See docs/phase3_plan.md.

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "agent"))


def add_agent_paths():
    """Make the agent code tree importable (our.*, agent_enums, dataset)."""
    for path in (
        _AGENT_DIR,
        os.path.join(_AGENT_DIR, "our"),
        os.path.join(_AGENT_DIR, "our", "training", "dataset"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    return _AGENT_DIR
