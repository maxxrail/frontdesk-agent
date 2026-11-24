"""Guardrails on what goes into the model and what comes out of it.

The model is not trusted, and neither is anything it reads. Three kinds of
text reach it: the client's SMS, policy documents (via search_policy), and
staff notes. Only the last is trusted, and its marker, a line starting
"[Staff note]", is something any client can type. So:

Input (``clean_untrusted``, ``inbound_too_long``):
- Unicode is NFKC-normalised and invisible format characters (zero-width
  spaces, bidi overrides) are removed, so "ibu​profen" or a fullwidth
  "［Staff note］" reach the risk rules and the model as what they look like.
- The staff-note marker in untrusted text is defanged to "(staff note)".
- A message over ``MAX_INBOUND_CHARS`` goes to a person without a model
  call: no client needs a page of text to book, and the model's attention is
  the thing an attacker pads.

Tool permissions never come from text. The set of tools is fixed before the
model runs, from the deterministic risk rules (agent/loop.py), and the one
destructive tool, cancel_appointment, holds for a person whenever policy
documents are in the model's context (agent/tools.py, ADR 0017).

Output (``check_reply``): the last thing before a text is sent. A reply is
held for a person, not sent, if it contains another client's full name or
phone number, payment card data, or clinical advice.
"""

import re
import unicodedata
import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Client, ReviewKind

# Twilio joins long messages up to 1600 characters (the column size). A
# real front-desk text is under 300; 1000 leaves room for a long complaint.
MAX_INBOUND_CHARS = 1000

_STAFF_MARKER = re.compile(r"\[\s*staff[\s_-]*note\s*\]", re.IGNORECASE)
_KEEP_CONTROL = {"\n", "\t"}


def clean_untrusted(text: str) -> str:
    """What the model and the risk rules see of untrusted text.

    The stored message keeps the raw text: staff see exactly what was sent.
    """
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue  # zero-width and bidi controls: invisible, so dropped
        if cat == "Cc" and ch not in _KEEP_CONTROL:
            out.append(" ")
            continue
        out.append(ch)
    return _STAFF_MARKER.sub("(staff note)", "".join(out))


def inbound_too_long(text: str) -> bool:
    return len(text) > MAX_INBOUND_CHARS


# -- output ----------------------------------------------------------------

# Phrases that are advice, not logistics. The agent may say "a team member
# will follow up" or "call 911"; it may not tell anyone what to take or do.
# The eval suite's universal check uses the same function (evaluation/checks.py),
# on drafts held for review as well as on sent texts.
MEDICAL_ADVICE = re.compile(
    "|".join(
        [
            r"\b\d+\s?mg\b",
            # Not "whether you can take Advil": that defers the question.
            r"(?<!can )\b(take|try|use) (some |an? )?(ibuprofen|advil|tylenol|"
            r"acetaminophen|aspirin|painkillers?|pain relievers?|antibiotics?)",
            r"\b(rinse|gargle) with",
            r"\bsalt ?water\b",
            r"\bapply (an? )?(ice|heat|cold|warm)",
            r"\b(ice|heat) (it|the area)\b",
            r"\byou (probably |likely |may |might )?have (an? )?(infection|dry socket|abscess|"
            r"fracture|sprain|tear|tendonitis)",
            r"\bsounds like (an? )?(infection|dry socket|abscess|sprain|tendonitis|normal)",
            r"\b(it'?s|that'?s|this is) (probably |likely |perfectly |completely )?"
            r"(normal|nothing to worry|fine)\b",
            r"\b(perfectly|completely|totally|quite|very) normal\b",
            r"\bnormal (after|for|with|to have)\b",
            r"\b(is|are) (safe|fine|ok|okay) to (take|eat|drink|run|exercise)",
            r"\b(stop|keep) (taking|doing) (the|your)",
        ]
    ),
    re.IGNORECASE,
)

# "Let me check whether Advil is okay to take" asks the question, it does not
# answer it (issue #24, risk-02). A match is excused when its clause opens
# with "whether", or with "if" after "ask" or "check". A bare "if" is not
# enough: "if it's sore it's normal" is advice. The clause ends at . ? ! , ;
# or "but"/"so", so "whether you ate or not, Advil is fine to take" and
# "not sure whether you need it but take some Advil" are still caught.
_CLAUSE = r"(?:(?!\b(?:but|so)\b)[^.?!,;])"
_DEFERS_THE_QUESTION = re.compile(
    rf"\b(whether|(ask|check)\w*\b{_CLAUSE}{{0,30}}\bif)\b{_CLAUSE}{{0,40}}$", re.IGNORECASE
)


def medical_advice(text: str) -> str | None:
    """The first phrase in ``text`` that gives clinical advice, or None."""
    for m in MEDICAL_ADVICE.finditer(text):
        if not _DEFERS_THE_QUESTION.search(text, 0, m.start()):
            return m.group(0)
    return None


# Asking for card details by SMS is the start of a PCI problem, even if the
# clinic meant well; payments go through the clinic's own channels.
_PAYMENT_REQUEST = re.compile(
    r"\b(card|credit card|debit card) (number|details|info)|\bcvv\b|\bcvc\b|"
    r"\bsecurity code\b|\bexpir(y|ation) date\b",
    re.IGNORECASE,
)
_DIGIT_RUN = re.compile(r"(?<![\d])\d(?:[ -]?\d){12,18}(?![\d])")
_PHONE = re.compile(r"(?<![\d+])\+?1?[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\d])")
_POSSESSIVE = re.compile(r"'s?$")
_WORD = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")


@dataclass(frozen=True, slots=True)
class Finding:
    kind: ReviewKind
    rule: str  # "other_client", "payment_data", "clinical_advice": for audit and metrics
    reason: str


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def payment_data(text: str) -> str | None:
    for m in _DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            return "a card number"
    if m2 := _PAYMENT_REQUEST.search(text):
        return f"a request for card details ('{m2.group(0)}')"
    return None


def phone_candidates(text: str) -> set[str]:
    """North American numbers in ``text``, in the E.164 form clients are stored in."""
    out = set()
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 10:
            out.add("+1" + digits)
        elif len(digits) == 11 and digits.startswith("1"):
            out.add("+" + digits)
    return out


def _words(text: str) -> list[str]:
    """Lowercased words, possessives cut: "Reyes's" is "reyes"."""
    return [_POSSESSIVE.sub("", w.lower()) for w in _WORD.findall(text.replace("’", "'"))]


async def other_client_details(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    text: str,
    client_said: str = "",
) -> str | None:
    """Another client's phone number, or first and last name side by side.

    A first name alone is not checked: providers, staff and the texting
    client share first names with someone in every clinic, and the tools
    cannot return another client's name in the first place. This is the
    backstop, not the wall.

    Details the client typed themselves are not a disclosure: "I can't share
    Morgan Reyes's appointment" repeats their words. The first paid eval run
    held three correct refusals like that before this rule (ADR 0017).
    """
    said_words = _words(client_said)
    said_pairs = set(zip(said_words, said_words[1:], strict=False))
    said_phones = phone_candidates(client_said)
    words = _words(text)
    phones = phone_candidates(text) - said_phones
    if len(words) < 2 and not phones:
        return None
    pairs = set(zip(words, words[1:], strict=False))
    lowered = sorted(set(words))
    conds: list[ColumnElement[bool]] = []
    if phones:
        conds.append(Client.phone.in_(phones))
    if pairs:
        conds.append(
            func.lower(Client.first_name).in_(lowered) & func.lower(Client.last_name).in_(lowered)
        )
    stmt = select(Client.first_name, Client.last_name, Client.phone).where(
        Client.tenant_id == tenant_id, Client.id != client_id, or_(*conds)
    )
    for first, last, phone in (await session.execute(stmt)).all():
        if phone in phones:
            return "another client's phone number"
        name = (first.lower(), last.lower())
        if name in pairs and name not in said_pairs:
            return "another client's name"
    return None


async def check_reply(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    reply: str,
    client_said: str = "",
) -> Finding | None:
    """The first problem with ``reply``, or None if it may be sent.

    ``client_said`` is the client's own texts in the conversation window.

    Clinical advice is checked first: it is the one that also sends the fixed
    medical safety message, because silence there is unsafe (ADR 0011).
    """
    if advice := medical_advice(reply):
        return Finding(ReviewKind.MEDICAL, "clinical_advice", f"clinical advice ('{advice}')")
    if what := payment_data(reply):
        return Finding(ReviewKind.ESCALATION, "payment_data", what)
    if what := await other_client_details(session, tenant_id, client_id, reply, client_said):
        return Finding(ReviewKind.ESCALATION, "other_client", what)
    return None
