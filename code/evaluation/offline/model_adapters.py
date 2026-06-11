"""Belief-model adapters for the offline replay harness (Workstream 0).

`BeliefBackend` is the protocol every evaluated model implements:

    name: str
    predict(events, ego_index, ego_role) -> {seat: P(evil)}   # 0-based seats
    predict_pairs(events, ego_index, ego_role) -> {(i, j): p} # optional

`events` is any chronological GameRecord-event prefix that does not split a
proposal/vote pair (see event_schema.py). Backends build their own inputs
from it — the factor_v2 backend projects onto the legacy 21-int vector, the
card backends consume proposal cards — so the harness stays model-agnostic.

Replay is vibes-off by definition (pure model, neutral priors); recorded-
vibes evaluation reads the committed CSV traces instead of calling a backend
(see replay_eval.py --vibes recorded).
"""

import contextlib
import io
import os

from . import add_agent_paths
from .metrics import pair_scores_from_marginals

AGENT_DIR = add_agent_paths()

from agent_enums import ATEAM  # noqa: E402
from our.proposal_cards import cards_from_events, v2_vector_from_cards  # noqa: E402


def _team_enum(ego_role):
    if isinstance(ego_role, ATEAM):
        return ego_role
    name = str(getattr(ego_role, "name", ego_role)).upper()
    if name == "GOOD":
        return ATEAM.GOOD
    if name == "EVIL":
        return ATEAM.EVIL
    raise ValueError(f"bad ego_role {ego_role!r}")


@contextlib.contextmanager
def _quiet_in_agent_dir():
    """The legacy model resolves 'our/models/...' relative to the cwd and
    prints progress lines; load and predict inside this context."""
    cwd = os.getcwd()
    os.chdir(AGENT_DIR)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        os.chdir(cwd)


class FactorV2Backend:
    """The deployed phase-2 model: FactorGraphModelV2 over the legacy 21-int
    state vector (resolved quests only). algorithm='max' matches the live
    agent's predict_probs call."""

    name = "factor_v2"

    def __init__(self, model_dir="v2/", algorithm="max"):
        self.algorithm = algorithm
        with _quiet_in_agent_dir():
            from our.model_reduced_categories import FactorGraphModelV2
            self.model = FactorGraphModelV2()
            self.model.construct()
            self.model.load_from_file(model_dir)
        self._cache = {}

    def predict(self, events, ego_index, ego_role):
        vector = v2_vector_from_cards(cards_from_events(events))
        team = _team_enum(ego_role)
        key = (tuple(vector), ego_index, team.name)
        if key not in self._cache:
            with _quiet_in_agent_dir():
                probs = self.model.predict_probs(
                    list(vector), self_role=team, self_index=ego_index,
                    algorithm=self.algorithm)
            self._cache[key] = {i: probs[i + 1]["evil"] for i in range(6)}
        return dict(self._cache[key])

    def predict_pairs(self, events, ego_index, ego_role):
        return pair_scores_from_marginals(
            self.predict(events, ego_index, ego_role))


class FactorV3Backend:
    """Track A: card-set encoder + exact exactly-2-evil enumeration
    (our/model_v3.py). sum-product is the calibrated runtime choice."""

    name = "factor_v3"

    def __init__(self, model_dir="v4_trackA/", algorithm="sum"):
        self.algorithm = algorithm
        with _quiet_in_agent_dir():
            from our.model_v3 import FactorGraphModelV3
            self.model = FactorGraphModelV3()
            self.model.load_from_file(model_dir)

    def predict(self, events, ego_index, ego_role):
        cards = cards_from_events(events)
        probs = self.model.predict_probs(
            cards, self_role=str(_team_enum(ego_role).name).lower(),
            self_index=ego_index, algorithm=self.algorithm)
        return {i: probs[i + 1]["evil"] for i in range(6)}

    def predict_pairs(self, events, ego_index, ego_role):
        cards = cards_from_events(events)
        return self.model.pair_posterior(
            cards, self_role=str(_team_enum(ego_role).name).lower(),
            self_index=ego_index)


class SeqV1Backend:
    """Track B: GRU over the same proposal cards with a 15-way pair head
    (our/seq_belief_model.py)."""

    name = "seq_v1"

    def __init__(self, model_dir="seq_v1/"):
        with _quiet_in_agent_dir():
            from our.seq_belief_model import SeqBeliefModel
            self.model = SeqBeliefModel()
            self.model.load_from_file(model_dir)

    def predict(self, events, ego_index, ego_role):
        cards = cards_from_events(events)
        probs = self.model.predict_probs(
            cards, self_role=str(_team_enum(ego_role).name).lower(),
            self_index=ego_index)
        return {i: probs[i + 1]["evil"] for i in range(6)}

    def predict_pairs(self, events, ego_index, ego_role):
        cards = cards_from_events(events)
        return self.model.pair_posterior(
            cards, self_role=str(_team_enum(ego_role).name).lower(),
            self_index=ego_index)


BACKENDS = {
    "factor_v2": FactorV2Backend,
    "factor_v3": FactorV3Backend,
    "seq_v1": SeqV1Backend,
}


def build_backend(name, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"unknown model '{name}' (have {sorted(BACKENDS)})")
    return BACKENDS[name](**kwargs)
