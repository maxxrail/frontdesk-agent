# Load test

50 synthetic clients text one clinic at the same time, through the real
Twilio webhook (signed requests), the Redis queue, the worker, the agent and
the outbox. The model is a free fake (`FDA_LLM_PROVIDER=fake`): one lookup and
one reply per turn, 0.9 s per model call, about what Haiku takes. Texts are
printed, never sent. Every number is in area code 999, which is unassigned.

## Run it

```
uv run python scripts/seed.py            # once; --reset only if you mean it
docker compose -f docker-compose.yml -f loadtest/compose.loadtest.yml up -d --build
uv run locust -f loadtest/locustfile.py --headless -u 50 -r 2 -t 3m --host http://localhost:8000
uv run python loadtest/report.py --since-minutes 3
```

Locust reports the webhook only: it stores the text and queues a job, then
answers Twilio. The SLO is about the reply, which `report.py` measures from
the database (inbound `created_at` to the reply's send time).

To kill Redis during a run: `docker compose stop redis`, then `start` it a
minute later. Watch `localhost:9090/alerts`.

## Results (2026-11-23, Windows 11 laptop, Docker Desktop)

| Run | Webhook requests | Failed | Webhook p99 / max | Replies | Reply p50 / p95 / max |
|---|---|---|---|---|---|
| Steady, 50 clients, 3 min | 303 | 0 | 75 ms / 140 ms | 301 of 303 sent [^1] | 3.02 s / 3.22 s / 3.27 s |
| Redis killed for 60 s, 4 min | 398 | 0 | 1.9 s / 3.8 s | 398 of 398 sent | 2.91 s / 3.24 s / 34.4 s |

A reply's 3 s is three fake model calls (a lookup drafted by the cheap model,
then its reply handed to the main one, ADR 0019) plus about 0.3 s of system.

With Redis down, the API runs jobs itself (`frontdesk_queue_fallback_total`)
and `DependencyDown{dependency="redis"}` goes from pending at 20 s to firing
by 60 s. The 34 s reply was a job that was in Redis when it died: it ran once
the worker came back, delayed but not lost. The worker's `TargetDown` is
pending meanwhile, because arq exits without Redis and compose restarts it.

[^1]: The two missing replies were still being handled when the report ran.

### What the first run found

The first run, before these fixes, answered every text but showed three bugs,
each now fixed and tested (commit "Fix three failures the first load test
found"):

- `ConsoleSender` gave every text the same provider id, so from the second
  text on the outbox row would not save and the sweeper re-sent it every
  minute.
- With Redis killed, single webhook requests hung 6-42 s (Twilio gives up at
  15 s). Redis attempts are now capped at 2 s and skipped for 10 s after a
  failure.
- The Docker image ignored `uv.lock` and installed sentry-sdk 3.0, which
  crashes at startup with our settings. It now builds from the lockfile.

## Against the real model

A short run against Claude, for the latencies the fake cannot show, is in
the Module 7 PR. It costs money, so it is run by hand and its cost printed.
