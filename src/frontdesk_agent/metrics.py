"""Prometheus metrics: what the dashboard and the SLO alerts read.

The API serves them at /metrics; the worker on its own port
(``FDA_METRICS_PORT``, 9101 in docker compose). Labels are bounded sets
(route templates, models, prompt versions, tools, outcomes) and never ids,
phone numbers or tenants, because /metrics is unauthenticated: in
production it is reachable only inside the network (Module 8).

Three numbers are read at scrape time instead of counted, because they are
state, not events: review queue depth (Postgres), whether Redis answers,
and the latest eval pass rate per prompt version (evals/results/*.json).
"""

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

# Seconds. An SMS turn is several model calls; the SLO line is 8 s.
_TURN_BUCKETS = (0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60)
_FAST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

HTTP_REQUESTS = Counter(
    "frontdesk_http_requests_total", "HTTP requests.", ["method", "route", "status"]
)
HTTP_SECONDS = Histogram(
    "frontdesk_http_request_seconds", "HTTP request latency.", ["route"], buckets=_FAST_BUCKETS
)
SMS_TURNS = Counter("frontdesk_sms_turns_total", "Inbound SMS handled, by outcome.", ["outcome"])
SMS_TURN_SECONDS = Histogram(
    "frontdesk_sms_turn_seconds",
    "Time to handle one inbound SMS, from job start to reply or hold.",
    buckets=_TURN_BUCKETS,
)
JOBS_FAILED = Counter("frontdesk_jobs_failed_total", "Jobs that raised.", ["job"])
QUEUE_FALLBACK = Counter(
    "frontdesk_queue_fallback_total", "Jobs run in the API process because Redis was down.", ["job"]
)
LLM_CALLS = Counter("frontdesk_llm_calls_total", "Model calls.", ["model", "prompt", "outcome"])
LLM_SECONDS = Histogram(
    "frontdesk_llm_call_seconds", "Model call latency.", ["model"], buckets=_FAST_BUCKETS
)
LLM_TOKENS = Counter("frontdesk_llm_tokens_total", "Model tokens.", ["model", "direction"])
LLM_COST = Counter("frontdesk_llm_cost_usd_total", "Model spend in USD.", ["model", "prompt"])
TURN_COST = Histogram(
    "frontdesk_agent_turn_cost_usd",
    "Model spend for one agent turn (one inbound SMS).",
    buckets=(0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064),
)
AGENT_ROUTES = Counter(
    "frontdesk_agent_routes_total",
    "Agent steps drafted by the lookup model: run as is, or handed to the main model.",
    ["route"],
)
TOOL_CALLS = Counter("frontdesk_tool_calls_total", "Agent tool calls.", ["tool", "outcome"])
REVIEW_QUEUE = Gauge("frontdesk_review_queue_depth", "Pending review items.", ["kind"])
DEPENDENCY_UP = Gauge(
    "frontdesk_dependency_up", "1 if the dependency answered at scrape time.", ["dependency"]
)
EVAL_PASS_RATE = Gauge(
    "frontdesk_eval_pass_rate",
    "Pass rate of the latest eval run for this prompt version and model.",
    ["prompt_version", "model"],
)


_OUTCOME = re.compile(r"^[a-z_]{1,40}(:[a-z_]{1,40})?$")


def sms_outcome(outcome: str) -> str:
    """Outcomes are code-made labels ('agent_reply', 'held_for_review:medical');
    anything else folds into 'other' so the label set stays bounded."""
    return outcome if _OUTCOME.fullmatch(outcome) else "other"


@contextmanager
def timed(histogram: Histogram, **labels: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        (histogram.labels(**labels) if labels else histogram).observe(time.perf_counter() - start)


def record_llm_call(
    model: str,
    prompt: str,
    *,
    seconds: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost: Decimal | None = None,
    ok: bool = True,
) -> None:
    LLM_CALLS.labels(model, prompt, "ok" if ok else "error").inc()
    LLM_SECONDS.labels(model).observe(seconds)
    if input_tokens:
        LLM_TOKENS.labels(model, "input").inc(input_tokens)
    if output_tokens:
        LLM_TOKENS.labels(model, "output").inc(output_tokens)
    if cache_read_tokens:
        LLM_TOKENS.labels(model, "cache_read").inc(cache_read_tokens)
    if cache_write_tokens:
        LLM_TOKENS.labels(model, "cache_write").inc(cache_write_tokens)
    if cost:
        LLM_COST.labels(model, prompt).inc(float(cost))


def load_eval_scores(results_dir: Path) -> int:
    """Set EVAL_PASS_RATE from the newest results file per (prompt, model).
    Returns how many pairs were found."""
    latest: dict[tuple[str, str], tuple[str, float]] = {}
    for path in results_dir.glob("*.json") if results_dir.is_dir() else []:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
            run, rate = r["run"], r["summary"]["pass_rate"]
            key = (str(run["prompt_version"]), str(run["model"]))
            started = str(run.get("started_at", ""))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # not a results file
        if key not in latest or started > latest[key][0]:
            latest[key] = (started, float(rate))
    for (prompt, model), (_, rate) in latest.items():
        EVAL_PASS_RATE.labels(prompt, model).set(rate)
    return len(latest)
