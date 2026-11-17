"""The two synthetic demo clinics, shared by the seed script and the evals.

Hours match each clinic's hours document in data/policies/<slug>/, so the
agent's availability and its policy answers agree.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Clinic:
    name: str
    slug: str
    business_type: str
    sms_number: str
    area_code: str
    providers: tuple[tuple[str, str], ...]
    # weekday (0 = Monday) -> (open, close), matching the clinic's hours document
    hours: dict[int, tuple[str, str]]


CLINICS = (
    Clinic(
        "Maple Dental",
        "maple-dental",
        "dental office",
        "+14165550100",
        "416",
        (("Dr. Amara Osei", "Dentist"), ("Lena Park", "Hygienist")),
        {
            0: ("08:00", "19:00"),
            1: ("08:00", "19:00"),
            2: ("08:00", "19:00"),
            3: ("08:00", "19:00"),
            4: ("08:00", "16:00"),
        },
    ),
    Clinic(
        "Northside Physio",
        "northside-physio",
        "physiotherapy clinic",
        "+16475550100",
        "647",
        (("Sam Ritter", "Physiotherapist"), ("Priya Nair", "Physiotherapist")),
        {
            0: ("07:00", "19:00"),
            1: ("10:00", "20:00"),
            2: ("07:00", "19:00"),
            3: ("10:00", "20:00"),
            4: ("07:00", "19:00"),
            5: ("08:00", "13:00"),
        },
    ),
)


def fake_phone(area_code: str, i: int) -> str:
    """NANP 555-0100..0199 numbers are reserved for fictional use."""
    return f"+1{area_code}555{110 + i:04d}"
