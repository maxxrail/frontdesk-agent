"""Loading and rendering versioned prompt files.

Prompts live in prompts/<name>/<version>.md with front matter and two
sections: `# System` and `# User`. Keeping them on disk means a prompt change
shows up as a reviewable diff.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


class PromptError(Exception):
    """A prompt could not be loaded or rendered."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """One version of one prompt."""

    name: str
    version: str
    system: str
    user: str

    def render(self, **values: str) -> tuple[str, str]:
        """Substitute placeholders in both sections.

        Raises if any placeholder is left unfilled, so a half-rendered prompt
        can never reach the model.
        """
        rendered = []
        for section in (self.system, self.user):
            out = _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), m.group(0))), section)
            missing = sorted(set(_PLACEHOLDER.findall(out)))
            if missing:
                raise PromptError(
                    f"{self.name}/{self.version}: missing values for {', '.join(missing)}"
                )
            rendered.append(out.strip())
        return rendered[0], rendered[1]


def _split_sections(body: str, name: str, version: str) -> tuple[str, str]:
    parts = re.split(r"^# (System|User)\s*$", body, flags=re.MULTILINE)
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].lower()] = parts[i + 1]
    if "system" not in sections or "user" not in sections:
        raise PromptError(f"{name}/{version}: needs both '# System' and '# User' sections")
    return sections["system"], sections["user"]


@lru_cache
def load_prompt(name: str, version: str, prompts_dir: Path | None = None) -> Prompt:
    """Load a prompt from disk. Cached, since prompt files do not change at runtime."""
    directory = prompts_dir or PROMPTS_DIR
    path = directory / name / f"{version}.md"
    if not path.is_file():
        raise PromptError(f"No prompt file at {path}")

    raw = _FRONT_MATTER.sub("", path.read_text(encoding="utf-8"))
    system, user = _split_sections(raw, name, version)
    return Prompt(name=name, version=version, system=system.strip(), user=user.strip())
