# model_factory.py
#
# Workstream 4: belief-model selection for the live agents, mirroring the
# load_policy_config env pattern. The A/B cells of phase 3 swap Good's
# detector without touching Evil's:
#
#   GRAIL_BELIEF_MODEL_GOOD / GRAIL_BELIEF_MODEL_EVIL  (per side)
#   GRAIL_BELIEF_MODEL                                 (both-side fallback)
#   default: factor_v2 (the deployed phase-2 model)
#
# Values: factor_v2 | factor_v3 | seq_v1, optionally "name:model_dir" to
# point at a non-default checkpoint directory under our/models/.
#
# build_belief_model returns (model, spec). spec tells ACLAgent what the
# model consumes:
#   input    "vector21" (legacy GameInfo.get_state_vector) or "cards"
#            (proposal_cards.cards_from_history on the live history)
#   algorithm  inference algorithm to pass to predict_probs

import os

from agent_enums import ATEAM


def _selected_name(team):
    side = "EVIL" if team == ATEAM.EVIL else "GOOD"
    return (
        os.environ.get(f"GRAIL_BELIEF_MODEL_{side}", "").strip()
        or os.environ.get("GRAIL_BELIEF_MODEL", "").strip()
        or "factor_v2"
    )


def vibes_disabled():
    """GRAIL_DISABLE_VIBES truthy -> skip the LLM-vibes prior update."""
    return os.environ.get("GRAIL_DISABLE_VIBES", "").strip().lower() in (
        "1", "true", "yes", "on")


def build_belief_model(team):
    """Construct + load the belief model selected for this side."""
    selection = _selected_name(team)
    name, _, model_dir = selection.partition(":")
    name = name.strip()

    if name == "factor_v2":
        from our.model_reduced_categories import FactorGraphModelV2
        model = FactorGraphModelV2()
        model.construct()
        model.load_from_file(model_dir or "v2/")
        spec = {"name": name, "input": "vector21", "algorithm": "max"}
    elif name == "factor_v3":
        from our.model_v3 import FactorGraphModelV3
        model = FactorGraphModelV3()
        model.load_from_file(model_dir or "v4_trackA/")
        spec = {"name": name, "input": "cards", "algorithm": "sum"}
    elif name == "seq_v1":
        from our.seq_belief_model import SeqBeliefModel
        model = SeqBeliefModel()
        model.load_from_file(model_dir or "seq_v1/")
        spec = {"name": name, "input": "cards", "algorithm": "sum"}
    else:
        raise ValueError(
            f"Unknown GRAIL belief model '{selection}' "
            "(expected factor_v2 | factor_v3 | seq_v1)")
    return model, spec
