# Tests for the phase-3 concluding persona/variant bank (our/prompts.py).
#
# Run from code/agent:  python3 test_prompt_personas.py   (or pytest)
#
# The load-bearing guarantees:
#   1. The default PromptHint templates are untouched by the new code.
#   2. Every persona template still .format()s with the exact kwargs ACLAgent
#      passes (a stray brace in a persona voice would crash the agent mid-game).
#   3. Persona templates keep the JSON-output contract and the strategic
#      wording; only the output_style paragraph changes.
#   4. assign_personas is deterministic, identical for every computing agent,
#      and gives the six players six distinct personas.

# Load prompts.py directly by path so we skip our/__init__.py (which imports
# torch, absent outside the Docker image) — same pattern as test_evil_policy.py.
import importlib.util
import os

_ppath = os.path.join(os.path.dirname(__file__), "our", "prompts.py")
_spec = importlib.util.spec_from_file_location("prompts_standalone", _ppath)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

PERSONA_STYLES = _prompts.PERSONA_STYLES
PersonaPromptHint = _prompts.PersonaPromptHint
PromptHint = _prompts.PromptHint
assign_personas = _prompts.assign_personas
build_prompt_hint = _prompts.build_prompt_hint

GOOD_KWARGS = dict(
    name="Sam",
    logs="Voiceover : hello",
    latest_probabilities={"Sam": 0.5},
    role="Servant-1",
    secret_info="",
    team_comp=["Sam", "Mia"],
    party_leader="Mia",
    quest_history="No prior Quests; this is the first Round.",
    quest_num=1,
    party_size=2,
    public_plan="",
    planned_vote="approve",
)

MESSAGE_TEMPLATES = (
    "generate_message_from_log_good",
    "generate_proposal_message_good",
    "confirm_proposal_message_good",
    "generate_message_from_log_evil",
    "generate_proposal_message_evil",
    "confirm_proposal_message_evil",
)


def test_default_templates_unchanged():
    # The class body was not edited; spot-check the exact legacy voice and the
    # exact strategic wording that the extension's docs cite.
    assert "as if you are a college student" in PromptHint.output_style
    assert PromptHint.output_style in PromptHint.generate_message_from_log_good
    assert "subtly support it without being obvious" in PromptHint.generate_message_from_log_evil
    assert "Your planned mechanical vote is to {planned_vote}" in PromptHint.generate_message_from_log_evil


def test_every_persona_formats_with_agent_kwargs():
    for key in PERSONA_STYLES:
        hint = build_prompt_hint(key)
        assert isinstance(hint, PersonaPromptHint) and hint.persona == key
        for attr in MESSAGE_TEMPLATES:
            rendered = getattr(hint, attr).format(**GOOD_KWARGS)
            assert "{" not in rendered.replace("{'Sam': 0.5}", ""), (key, attr)
        # vibes template uses doubled braces for its example output
        hint.get_vibes_player_agreement.format(**GOOD_KWARGS)


def test_personas_change_only_the_voice():
    for key, style in PERSONA_STYLES.items():
        hint = build_prompt_hint(key)
        for attr in MESSAGE_TEMPLATES:
            default_t = getattr(PromptHint, attr)
            persona_t = getattr(hint, attr)
            # style swapped in, default voice gone
            assert style in persona_t, (key, attr)
            assert PromptHint.output_style not in persona_t, (key, attr)
            # everything around the style is identical
            assert persona_t == default_t.replace(PromptHint.output_style, style)
            # the JSON contract and strategic wording survive
            assert "JSON object with one key named 'message'" in persona_t
        # evil strategy content untouched by the persona voice
        assert "maintaining your cover as a good player" in hint.generate_message_from_log_evil
        assert "{planned_vote}" in hint.generate_message_from_log_evil
        # vibes/belief template is byte-identical (no output_style inside)
        assert hint.get_vibes_player_agreement == PromptHint.get_vibes_player_agreement


def test_assignment_deterministic_and_distinct():
    names = ["Sam", "Paul", "Luca", "Jane", "Kira", "Mia"]
    a = assign_personas("WTLF", names)
    b = assign_personas("WTLF", list(reversed([n.upper() for n in names])))
    assert a == b  # order/case independent -> every agent agrees
    assert set(a) == {n.lower() for n in names}
    assert len(set(a.values())) == 6  # six distinct speakers
    assert set(a.values()) == set(PERSONA_STYLES)
    # different game -> (almost surely) different assignment, still valid
    c = assign_personas("XQZR", names)
    assert set(c.values()) == set(PERSONA_STYLES)


if __name__ == "__main__":
    test_default_templates_unchanged()
    test_every_persona_formats_with_agent_kwargs()
    test_personas_change_only_the_voice()
    test_assignment_deterministic_and_distinct()
    print("test_prompt_personas: all tests passed")
