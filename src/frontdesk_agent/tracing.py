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
from frontdesk_agent.llm.pricing import cost_usd

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
        with _llm_span("complete", prompt_name, prompt_version) as span:
            out = await self._inner.complete(
                system=system,
                user=user,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                max_tokens=max_tokens,
            )
            _record(span, out.model, out.input_tokens, out.output_tokens, out.stop_reason)
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
        with _llm_span("converse", prompt_name, prompt_version) as span:
            out = await self._inner.converse(
                system=system,
                messages=messages,
                tools=tools,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                max_tokens=max_tokens,
            )
            _record(span, out.model, out.input_tokens, out.output_tokens, out.stop_reason)
            span.set_attribute("llm.tool_calls", len(out.tool_calls))
            return out


@contextmanager
def _llm_span(operation: str, prompt_name: str, prompt_version: str) -> Iterator[Span]:
    start = time.perf_counter()
    with tracer.start_as_current_span("llm.call", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("llm.prompt", f"{prompt_name}/{prompt_version}")
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            span.set_attribute("llm.latency_ms", round((time.perf_counter() - start) * 1000))


def _record(
    span: Span, model: str, input_tokens: int, output_tokens: int, stop: str | None
) -> None:
    attrs: dict[str, Any] = {
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "gen_ai.response.finish_reason": stop or "",
    }
    cost = cost_usd(model, input_tokens, output_tokens)
    if cost is not None:
        attrs["llm.cost_usd"] = float(cost)
    span.set_attributes(attrs)
