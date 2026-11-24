"""The risk policy, in code.

Some messages must reach a person before any reply goes out, however capable
the model is. Deterministic rules run on every inbound message *before* the
model sees it, so a model that misjudges risk cannot skip review:

- Money: refunds, disputes, balances, fee waivers. Prices are not "money":
  "how much is a cleaning" is a policy question the documents answer.
- Medical: symptoms, medication, clinical advice.
- A child's booking: a parent booking, moving or cancelling for their child
  is held as an escalation, because the tools book only the person texting
  (ADR 0022).

Two more rules live where the action happens, not here: a cancellation inside
24 hours is held by the cancel tool, and the model can escalate anything else
with the escalate_to_human tool.

Keyword rules are blunt on purpose. A false positive costs a human a minute;
a false negative sends unreviewed medical advice by text. evals/risk_rules.py
measures both on labelled messages, and tests/test_risk_rules_quality.py
fails if either gets worse. Tune on evals/golden/risk.jsonl only: the
held-out set is what keeps the numbers honest. On it the rules still miss
about one medical message in five, which is why the model's own
escalate_to_human is the second line, and why it is measured too (ADR 0014).
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
    # Added in Module 5 from the tuning set (evals/golden/risk.jsonl). Past
    # tense ("charged", "billed") is a complaint; "do you charge" is a question.
    r"\bcharged\b",
    r"\bbilled\b",
    r"\bdeclined\b",
    r"money back|\breverse|\breversal",
    r"from my (card|account|visa|mastercard|credit card)",
    r"(got|been|was|sent me an?) invoic",
    r"(not|n'?t|refuse to) (have to )?pay\b",
    r"payment plan|pay (it |you )?later|split (it|the (bill|payment))",
    r"(take|took|remove|drop|cancel) (the |that |this |my )?\S*\s?fee",
    r"\bfee\b.{0,25}\b(ridiculous|unfair|off\b)",
    r"(get|give me|have) a discount",
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
    r"\bnumb(?!er)",  # not "number"
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
    # Added in Module 5 from the tuning set (evals/golden/risk.jsonl).
    # Pain words in either order, and plain words for hurting.
    r"\bpain (is |has )?(been )?(so |really |very )?"
    r"(severe|bad|worse|unbearable|throbbing|sharp|constant|spread|moved|still)",
    r"(for|with|in|chest|much) (the |so much |a lot of )?pain\b",
    r"pain ?killers?|\bmeds\b|\bpills?\b|blood thinner",
    r"\bhurts?\b|\bhurting\b|\bach(e|es|ing)\b|\bthrobb|\bsensitive\b",
    r"\bdizz|\bfaint|\bnause|\bvomit|feel(ing)? (sick|unwell|ill)\b|\brash\b|\bhives\b",
    r"\bpregnan|\bdiabet|heart (valve|condition)|pacemaker",
    r"\bswallow",
    # Injury verbs only next to a tooth or body part: "my car broke down" is not medical.
    r"\b(tooth|teeth|crown|filling|implant|bracket|wire|stitches|jaw|knee|ankle|shoulder|"
    r"back|neck|hip|wrist)\b.{0,25}\b(crack|broke|chip|knocked|fell (out|off)|"
    r"came (out|off)|popped|gave (out|way)|lock)",
    r"\b(cracked|broke|chipped|knocked out|lost) (a |my |the )?(tooth|teeth|crown|filling)",
    r"is it (safe|ok|okay|alright|fine) to\b",
    r"can i (eat|drink|smoke|exercise|run|swim|work out|brush|floss)\b",
    r"\bgauze\b|dry socket|\blump\b|\bbump\b",
    r"should (i|we) (be )?worr|is (that|this|it) from the",
    r"since (the|my) (session|treatment|surgery|extraction|procedure|injection|"
    r"filling|implant|crown|sedation|massage)",
]

# A parent or guardian booking, moving or cancelling for their child (issue
# #22). Allowed, but staff set it up: the tools book only the person texting.
# Three prompt versions did not make the model escalate it reliably, so it is
# a rule (ADR 0022). It needs the child to be the one booked: "cancel, my son
# is sick" or "can I bring my kids" are the parent's own appointment.
_AGE = r"\d{1,2}[- ]?(?:year|yr)s?[- ]?olds?|\d{1,2} ?yo"
_KID = (
    r"sons?|daughters?|kids?|child|children|boys?|girls?|toddlers?|bab(?:y|ies)|teens?"
    r"|teenagers?|twins|step(?:son|daughter)s?|grand(?:son|daughter|kid|child)s?|grandchildren"
)
_THEIR = rf"\b(?:my|our) (?:(?:little|baby|two|three|both|{_AGE}) )*(?:{_KID}|{_AGE})\b"
_VISIT = r"appointments?|appts?|cleanings?|check-?ups?|exams?|visits?|sessions?|assessments?"
_CHILD = [
    # "book my son", "fit our toddler in", "sign my son up", "change my kids' appointments"
    rf"\b(?:book|schedule|sign|fit|squeeze|move|reschedule|cancel|change|enrol|enroll|register)"
    rf"(?: up| in)? {_THEIR}",
    rf"\bget {_THEIR} (?:booked|in|seen|scheduled)",
    # "anything friday for my kid", "an appointment for my son"
    rf"\b(?:book\w*|schedul\w*|{_VISIT}|slots?|opening|anything|room|space|time)\b[^.?!]{{0,40}}"
    rf"\bfor {_THEIR}",
    # "my daughter's cleaning", "my kids' appointments", "my baby girl's first"
    rf"{_THEIR}'s?'? (?:{_VISIT}|booking|first)",
    # "my 4 year old needs her first dental visit", "can my daughter have the 10am".
    # Not "can my 10 year old come in alone?": that asks the policy (policy-11).
    rf"{_THEIR} (?:needs|wants|has to|should have) (?:\w+ ){{0,3}}(?:{_VISIT})",
    rf"\bcan {_THEIR} (?:have|take|get|book)\b",
]

_MONEY_RE = re.compile("|".join(_MONEY), re.IGNORECASE)
_MEDICAL_RE = re.compile("|".join(_MEDICAL), re.IGNORECASE)
_CHILD_RE = re.compile("|".join(_CHILD), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RiskFlag:
    kind: ReviewKind
    matched: str

    @property
    def reason(self) -> str:
        what = {
            ReviewKind.MEDICAL: "a medical question",
            ReviewKind.MONEY: "a money matter",
        }.get(self.kind, "a booking for the client's child")
        return f"Client message looks like {what} (matched '{self.matched}')."


def classify(text: str) -> RiskFlag | None:
    """Medical wins over money: it is the one that cannot wait. A child's
    booking comes last: a sick child is a medical message first."""
    if m := _MEDICAL_RE.search(text):
        return RiskFlag(ReviewKind.MEDICAL, m.group(0))
    if m := _MONEY_RE.search(text):
        return RiskFlag(ReviewKind.MONEY, m.group(0))
    if m := _CHILD_RE.search(text):
        return RiskFlag(ReviewKind.ESCALATION, m.group(0))
    return None


# Sent at once for medical holds only, because silence there is unsafe. Fixed
# text, never model-written. Every other hold sends nothing until a decision.
MEDICAL_HOLDING_MESSAGE = (
    "Thanks for your message. A team member will reply shortly. If this is an "
    "emergency, call 911 or go to the nearest emergency department."
)

LATE_CANCEL_WINDOW_HOURS = 24
