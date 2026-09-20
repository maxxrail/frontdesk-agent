"""Prompt loading and rendering tests."""

import pytest

from frontdesk_agent.prompts import PromptError, load_prompt


def test_loads_reminder_prompt() -> None:
    prompt = load_prompt("reminder_draft", "v1")

    assert prompt.name == "reminder_draft"
    assert prompt.version == "v1"
    assert "{{business_name}}" in prompt.system
    assert "{{client_first_name}}" in prompt.user


def test_render_substitutes_every_placeholder() -> None:
    prompt = load_prompt("reminder_draft", "v1")

    system, user = prompt.render(
        business_name="Maple Dental",
        business_type="dental office",
        client_first_name="Dana",
        appointment_human="Tuesday, October 6 at 2:30 PM",
        provider_name="Dr. Osei",
        location="120 Main St",
    )

    assert "{{" not in system
    assert "{{" not in user
    assert "Maple Dental" in system
    assert "Dana" in user


def test_render_fails_loudly_when_a_value_is_missing() -> None:
    prompt = load_prompt("reminder_draft", "v1")

    with pytest.raises(PromptError, match="missing values"):
        prompt.render(business_name="Maple Dental", business_type="dental office")


def test_unknown_prompt_raises() -> None:
    with pytest.raises(PromptError, match="No prompt file"):
        load_prompt("does_not_exist", "v1")
