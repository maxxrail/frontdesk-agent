# Evals

Everything that measures the agent. Each script is runnable alone, prints its
cost, and stops at a hard spending cap.

| Script | What it measures | Cost |
|---|---|---|
| `run.py` | The suite: 132 cases (40 red team) through the production inbound path, an LLM judge, 30 simulated conversations, the policy-answer eval | about $1.70 a run, cap $3 |
| `risk_rules.py` | Keyword risk rules on labelled messages (tuning and held-out sets) | free |
| `retrieval.py` | Chunking and search modes on `golden/retrieval.jsonl` | free |
| `policy_answers.py` | Grounded policy answers on 60 questions | a few cents |
| `judge_labels.py` | The judge against hand labels (below) | about $0.12 |
| `history.py` | Scores per run as CSV, for charting across prompt versions | free |

CI (`.github/workflows/evals.yml`) runs `run.py` on pull requests that touch
the prompts, the agent, the model setting or the evals, posts the score table
on the PR, and compares it with `baseline.json`.

## The gate, and how noisy one run is

Each case runs once per suite run, against a real model, so results move
between identical runs. Two agent/v2 runs on PR #4, same prompt, model and
code, scored **80.3%** and **82.6%**, and **19 of 132 cases flipped** (8 from
pass to fail, 11 the other way). A single case changing is not a signal.

`thresholds.json` is set from that:

- overall pass rate at least **75%**, about twice the observed noise below
  the lower run;
- a floor per feature about 10 points below the lower run (booking has only
  22 cases, so one case is 4.5 points);
- at most **12 regressions** against `baseline.json` (8 came from noise
  alone);
- zero tolerance, one case is enough: a leak, a change to another client's
  booking, medical advice, a crashed case.

`baseline.json` is the second of those runs (every reply judged). Replace it
when a change is merged on purpose, from the CI artifact of that PR's run.

## Judge agreement with hand labels

The judge (`prompts/judge/v1`, Claude Sonnet 5.5, thinking off) grades
replies for **tone** (pass/fail) and **faithfulness** to the retrieved policy
(pass/fail/n/a). To check it, Maxx labelled 40 replies from the agent/v2 run on
PR #4 by hand, using the rubric word for word and without seeing the judge's
verdicts (`judge/labels.csv`). The judge then graded the same 40
(`judge/agreement.json`, 2026-11-16, $0.12):

| | Agreement | Cohen's kappa | n |
|---|---|---|---|
| Faithfulness | **95%** | **0.90** | 40 |
| Tone | **82.5%** | **0.38** | 40 |

**Faithfulness is trustworthy.** The two disagreements are rubric edge cases:
a hedged "we may not have evening hours" (labelled n/a, judged as a hours
claim), and "no fee over 24 hours", which the labeller read as following from
the excerpts and the judge did not.

**Tone is not yet.** The judge is stricter than the labeller: in 6 of 7
disagreements it failed a reply the labeller passed, mostly replies to pain or
diagnosis questions it called dismissive or evasive. Labels have few tone
failures (4 of 40), which also keeps kappa low. Treat the tone rate as a
trend between runs on the same judge, not an absolute; it is not gated.
Next: calibrate the rubric on these disagreements, then re-label a fresh 40.

**A measurement bug this caught.** The first agreement run judged only 24 of
the 40 rows and reported 96% tone agreement. Sonnet 5.5 thinks by default and
its thinking counts against `max_tokens`; at 300 tokens, verdicts came back
cut off. The same bug left 52 of 129 replies unjudged in the CI runs on PR #4,
so the judge rows in those PR comments cover only part of the replies. The
judge now runs with thinking off and 1024 tokens, and a cut-off verdict is
reported as such.
