"""The risk policy, in code.

Some messages must reach a person before any reply goes out, however capable
the model is. Deterministic rules run on every inbound message *before* the
model sees it, so a model that misjudges risk cannot skip review:

- Money: refunds, disputes, balances, fee waivers. Prices are not "money":
  "how much is a cleaning" is a policy question the documents answer.
- Medical: symptoms, medication, clinical advice.

Two more rules live where the action happens, not here: a cancellation inside
24 hours is held by the cancel tool, and the model can escalate anything else
with the escalate_to_human tool.

Keyword rules are blunt on purpose. A false positive costs a human a minute;
a false negative sends unreviewed medical advice by text. Module 5 measures
both rates on labelled messages.
"""

import re
from dataclasses import dataclass

from frontdesk_agent.db.models import ReviewKind

_MONEY = [
    r"refund",
    r"reimburs",
    r"over ?charg",
    r"charged me",
    r"double[- ]charged",
    r"charged twice",
    r"dispute",
    r"charge ?back",
    r"\bbalance\b",
    r"\bi owe\b",
    r"\bowe you\b",
    r"collections?\b",
    r"\bwaive",
    r"bill(ed|ing)? (is |was )?(wrong|incorrect|twice)",
    r"wrong (bill|charge|amount)",
    r"payment (failed|declined|didn.?t go through)",
    r"\bcredit (back|my|to my)\b",
]

_MEDICAL = [
    r"bleed",
    r"\bswell",
    r"\bswollen\b",
    r"infect",
    r"\bpus\b",
    r"\bfever\b",
    r"\b(severe|bad|worse|unbearable|throbbing|sharp|constant) "
    r"(tooth|back|knee|neck|jaw|shoulder)? ?pain",
    r"pain (is )?(getting |gotten )?worse",
    r"\bnumb",
    r"allergic",
    r"side effects?",
    r"antibiotic",
    r"ibuprofen|advil|tylenol|acetaminophen|aspirin|codeine|amoxicillin|naproxen|aleve",
    r"\bmedication",
    r"prescri(be|ption)",
    r"\bdos(e|age)\b",
    r"should i (take|stop|use|ice|heat)",
    r"is (it|this) normal",
    r"\bstitches\b",
    r"can'?t breathe|trouble breathing|hard to swallow",
]

_MONEY_RE = re.compile("|".join(_MONEY), re.IGNORECASE)
_MEDICAL_RE = re.compile("|".join(_MEDICAL), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RiskFlag:
    kind: ReviewKind
    matched: str

    @property
    def reason(self) -> str:
        what = "a medical question" if self.kind is ReviewKind.MEDICAL else "a money matter"
        return f"Client message looks like {what} (matched '{self.matched}')."


def classify(text: str) -> RiskFlag | None:
    """Medical wins over money: it is the one that cannot wait."""
    if m := _MEDICAL_RE.search(text):
        return RiskFlag(ReviewKind.MEDICAL, m.group(0))
    if m := _MONEY_RE.search(text):
        return RiskFlag(ReviewKind.MONEY, m.group(0))
    return None


# Sent at once for medical holds only, because silence there is unsafe. Fixed
# text, never model-written. Every other hold sends nothing until a decision.
MEDICAL_HOLDING_MESSAGE = (
    "Thanks for your message. A team member will reply shortly. If this is an "
    "emergency, call 911 or go to the nearest emergency department."
)

LATE_CANCEL_WINDOW_HOURS = 24
