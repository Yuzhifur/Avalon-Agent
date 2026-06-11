"""Dependency-free enums shared by the agent runtime and offline tooling.

These previously lived in agent_base.py, which also imports the OpenAI client
and TypeChat. The belief models, policies, and the offline replay harness only
need the enums, so they import from here and stay usable in environments
without LLM dependencies. agent_base re-exports them, so enum identity is the
same through either import path.
"""

from enum import Enum


class LLM(Enum):
    LOCAL = 1
    GPT = 2
    DEEPSEEK = 3


class ATEAM(Enum):
    GOOD = 1
    EVIL = 2


class AROLE(Enum):
    MERLIN = 1
    PERCIVAL = 2
    SERVANT = 3
    MORGANA = 4
    ASSASSIN = 5
    EVIL = 6
