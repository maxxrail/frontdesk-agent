# 8. Retrieval design chosen from measurements

- Status: accepted
- Date: 2026-11-14

## Context

Policy answers are only as good as the chunks retrieved for them. Chunk
strategy, chunk size and search mode are usually picked from a tutorial. This
module picks them from numbers instead.

Setup: 35 synthetic policy documents for two clinics, a golden set of 60
questions written the way clients text (50 answerable, 10 not, two of which
are only answerable from the *other* clinic's documents), and
`evals/retrieval.py`, which scores every combination in a rolled-back
transaction. Labels are evidence phrases rather than chunk IDs, so one set of
labels scores every chunker fairly.

Embeddings come from WordLlama, a static 256-dimension model whose weights
ship in its pip package: free, offline, CPU-only, and ~3 ms per query
end to end. The build environment could not reach model hubs, and the same
constraint keeps the service free to run. It is a weak model, which makes the
measurements more informative, not less.

## Results

Full table: `evals/reports/retrieval-20261114.md`. The rows that decided it:

| Chunker | Mode | Recall@5 | Recall@300 words | Words in top 5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| section:160/30 | vector | 0.95 | 0.98 | 141 | 0.876 |
| section:160/30 | hybrid | 0.91 | 0.96 | 164 | 0.747 |
| section:160/30 | keyword | 0.83 | 0.91 | 160 | 0.633 |
| fixed:160/30 | vector | 0.99 | 0.99 | 411 | 0.930 |
| section:40/10 | vector | 0.92 | 0.98 | 120 | 0.872 |

Three findings:

1. **Big fixed chunks only look best.** fixed:160 wins recall@5, but each
   clinic has ~18 such chunks, so the top 5 is a quarter of the corpus, and
   it hands the model 3x the text. At an equal 300-word budget every
   configuration scores 0.96 to 0.99: within one or two questions of 50.
2. **Hybrid search did not help.** Reciprocal rank fusion with Postgres
   full-text search never beat vector search on recall or MRR in any of six
   chunk configurations. The misses are conversational phrasings with little
   word overlap ("what happens if i just dont show up"), which keyword search
   cannot find, and equal-weight fusion then demotes vector's good rankings.
3. **Similarity cannot refuse.** The cut-off that stops all 10 unanswerable
   questions also stops 22% of answerable ones: "running 10 min behind"
   scores 0.14 while "laser gum surgery" scores 0.28.

## Decision

- **Chunking: section, 160 words, 30 overlap.** Equal to the alternatives at
  equal context cost, and a chunk is always one topic, so a citation names
  "Late cancellation fee" rather than a span of three policies. Most sections
  fit in one chunk; the overlap only matters for long FAQ sections.
- **Search mode: vector** (`FDA_RETRIEVAL_MODE`). Keyword and hybrid stay
  implemented, tested and one setting away, because a stronger embedder or
  exact-code queries (CDCP, WSIB) could change the result; rerun the eval
  before switching.
- **Refusal is the model's job, enforced in code** (ADR 0009). The similarity
  gate is set at 0.12, below every answerable question, and only stops
  gibberish from costing a model call.
- `tests/test_retrieval_quality.py` fails CI if recall@5 drops below 0.90 or
  MRR below 0.82 (today: 0.95 and 0.876).

## Consequences

- 50 answerable questions is small: 0.02 is one question. Differences under
  ~0.04 are noise, and the ADR treats them that way.
- The golden set was written by the same person as the documents, which
  flatters every configuration. Module 5 adds questions from real-style
  conversations; Module 9's pilot would add real ones.
- Rankings are deterministic (ties broken by document and position, never by
  UUID), so the report is reproducible run to run, verified by hashing three
  runs.
- pgvector applies the tenant filter after the HNSW scan. `ef_search` is
  raised to 100 so a tenant with few chunks among many still gets k results;
  with pgvector 0.8 (the CI image) iterative scans are the better fix once
  the corpus is large.
- Changing the embedder or chunker re-ingests automatically: each document
  stores the chunker label and model name its chunks were built with.
