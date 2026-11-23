"""OpenTelemetry traces: one SMS is one trace.

    POST /v1/channels/sms            (the webhook request)
    └── job handle_inbound           (the worker, joined via the job's arguments)
        └── agent.turn
            ├── llm.call             (one per model call: model, tokens, cost)
            ├── tool check_availability
            ├── tool search_policy
            │   └── retrieval.search
            └── llm.call

Spans are made by hand at these points rather than by auto-instrumentation
packages, so every attribute is chosen: ids, counts, model names, tokens,
costs and outcomes. Never message text, phone numbers or names. The job and
agent spans carry ``conversation.id``, so a trace backend finds every trace
for a conversation by that tag; log lines carry ``trace_id`` to jump the
other way.

Without ``FDA_OTEL_ENDPOINT`` no exporter is set up and the spans cost almost
nothing. Locally, ``docker compose up -d jaeger`` and
``FDA_OTEL_ENDPOINT=http://localhost:4318`` show them at localhost:16686.
"""

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

__all__ = [
    "SpanKind",
    "Status",
    "StatusCode",
    "TracedLLMClient",
    "init_tracing",
    "job_span",
    "tracer",
]

from frontdesk_agent import __version__
from frontdesk_agent.llm.pricing import Usage, cost_of
from frontdesk_agent.metrics import record_llm_call

if TYPE_CHECKING:
    from frontdesk_agent.config import Settings
    from frontdesk_agent.llm.base import LLMClient, LLMCompletion
    from frontdesk_agent.llm.messages import AgentTurn, Message, ToolSpec

tracer = trace.get_tracer("frontdesk_agent", __version__)

_provider: TracerProvider | None = None


def init_tracing(settings: "Settings", *, exporter: SpanExporter | None = None) -> bool:
    """Export spans if an endpoint (or, in tests, an exporter) is given.

    OpenTelemetry allows one global provider per process, so a second call
    only adds the exporter to the first provider.
    """
    global _provider
    endpoint = settings.otel_endpoint.rstrip("/")
    if not endpoint and exporter is None:
        return False
    if _provider is None:
        _provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.service_name,
                    "service.version": __version__,
                    "deployment.environment": settings.env,
                }
            )
        )
        trace.set_tracer_provider(_provider)
    if exporter is not None:
        _provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        _provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
    return True


def trace_id_hex(span: Span | None = None) -> str | None:
    ctx = (span or trace.get_current_span()).get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def carrier() -> dict[str, str]:
    """The current trace context, to send along with a queued job."""
    out: dict[str, str] = {}
    propagate.inject(out)
    return out


@contextmanager
def job_span(job: str, trace_carrier: Mapping[str, str] | None) -> Iterator[Span]:
    """A job's span, continuing the trace of whatever queued it."""
    parent = propagate.extract(dict(trace_carrier or {}))
    with tracer.start_as_current_span(f"job {job}", context=parent, kind=SpanKind.CONSUMER) as s:
        s.set_attribute("job.name", job)
        yield s


def set_conversation(conversation_id: str) -> None:
    trace.get_current_span().set_attribute("conversation.id", conversation_id)


class TracedLLMClient:
    """Wraps any LLMClient: each model call becomes an ``llm.call`` span with
    the model, prompt, tokens, latency and cost. Prompts and replies are not
    recorded; ``llm_calls`` in Postgres holds the numbers too."""

    def __init__(self, inner: "LLMClient") -> None:
        self._inner = inner

    async def complete(
        self,
        *,
        system: str,
        user: str,
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> "LLMCompletion":
        with _llm_span("complete", prompt_name, prompt_version) as c:
            out = await self._inner.complete(
                system=system,
                user=user,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                max_tokens=max_tokens,
            )
            _record(c, out, out.stop_reason)
            return out

    async def converse(
        self,
        *,
        system: str,
        messages: "list[Message]",
        tools: "list[ToolSpec]",
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> "AgentTurn":
        with _llm_span("converse", prompt_name, prompt_version) as c:
            out = await self._inner.converse(
                system=system,
                messages=messages,
                tools=tools,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                max_tokens=max_tokens,
            )
            _record(c, out, out.stop_reason)
            c.span.set_attribute("llm.tool_calls", len(out.tool_calls))
            return out


@dataclass
class _Call:
    span: Span
    prompt: str
    start: float


@contextmanager
def _llm_span(operation: str, prompt_name: str, prompt_version: str) -> Iterator[_Call]:
    prompt = f"{prompt_name}/{prompt_version}"
    with tracer.start_as_current_span("llm.call", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("llm.prompt", prompt)
        c = _Call(span, prompt, time.perf_counter())
        try:
            yield c
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            record_llm_call("unknown", prompt, seconds=time.perf_counter() - c.start, ok=False)
            raise


def _record(c: _Call, out: Usage, stop: str | None) -> None:
    """Span attributes and Prometheus numbers for one call, from the same
    values, so metrics work whether or not traces are exported."""
    seconds = time.perf_counter() - c.start
    cost = cost_of(out)
    attrs: dict[str, Any] = {
        "gen_ai.request.model": out.model,
        "gen_ai.usage.input_tokens": out.input_tokens,
        "gen_ai.usage.output_tokens": out.output_tokens,
        "gen_ai.usage.cache_read_input_tokens": out.cache_read_tokens,
        "gen_ai.usage.cache_creation_input_tokens": out.cache_write_tokens,
        "gen_ai.response.finish_reason": stop or "",
        "llm.latency_ms": round(seconds * 1000),
    }
    if cost is not None:
        attrs["llm.cost_usd"] = float(cost)
    c.span.set_attributes(attrs)
    record_llm_call(
        out.model,
        c.prompt,
        seconds=seconds,
        input_tokens=out.input_tokens,
        output_tokens=out.output_tokens,
        cache_read_tokens=out.cache_read_tokens,
        cache_write_tokens=out.cache_write_tokens,
        cost=cost,
    )
