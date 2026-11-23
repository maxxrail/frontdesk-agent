"""50 clients texting the clinic at once, through the real Twilio webhook.

Each Locust user is one synthetic client (a NANP 555-01xx number, reserved
for fiction) holding one conversation. Requests are signed exactly as Twilio
signs them, so they take the production path: signature check, store,
enqueue, worker, agent, outbox. See loadtest/README.md for how to run it.

Locust measures only the webhook (store and enqueue). The SLO is about the
reply, which the worker sends later; loadtest/report.py measures that from
the database after the run.
"""

import itertools
import os
import random
import uuid

from locust import HttpUser, between, task
from twilio.request_validator import RequestValidator

AUTH_TOKEN = os.environ.get("LOADTEST_AUTH_TOKEN", "loadtest-not-a-secret")
# The URL Twilio would call: the API's FDA_PUBLIC_BASE_URL plus the route.
PUBLIC_URL = os.environ.get("LOADTEST_PUBLIC_URL", "http://localhost:8000")
CLINIC = os.environ.get("LOADTEST_TO", "+14165550100")  # Maple Dental (seed.py)
ROUTE = "/v1/channels/sms"

MESSAGES = (
    "hi, do you have anything tomorrow morning?",
    "what time do you open on friday?",
    "can I move my appointment?",
    "what's your cancellation policy?",
    "do you take walk-ins?",
    "thanks!",
)

# Area code 999 is unassigned, so no load-test text can ever be addressed to
# a real person. Each run starts at a random block: reusing the last run's
# numbers would trip the per-phone limit and measure that instead.
_start = random.randrange(0, 9000, 100)
_numbers = (f"+1999555{n:04d}" for n in itertools.count(_start))


class TextingClient(HttpUser):
    # A person texts every 20-40 s: 15-30 texts in 10 minutes would trip the
    # per-phone limit (12 per 10 min, ADR 0018) and measure that instead.
    wait_time = between(20, 40)

    def on_start(self) -> None:
        self.phone = next(_numbers)
        self.validator = RequestValidator(AUTH_TOKEN)

    @task
    def text_the_clinic(self) -> None:
        form = {
            "To": CLINIC,
            "From": self.phone,
            "Body": random.choice(MESSAGES),
            "MessageSid": f"SMload{uuid.uuid4().hex[:28]}",
        }
        signature = self.validator.compute_signature(PUBLIC_URL + ROUTE, form)
        self.client.post(ROUTE, data=form, headers={"X-Twilio-Signature": signature}, name=ROUTE)
