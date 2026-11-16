# Retrieval report, 2026-11-14

Golden set: 60 questions (50 answerable, 10 unanswerable), embedder `wordllama-l2_supercat-256`, k = 5. Recall and MRR are over answerable questions only.

Recall@300w gives every configuration the same 300 words of retrieved context, the fair comparison between chunk sizes. Words@5 is how much text the top 5 hands the model.

| Chunker | Mode | Chunks | Recall@5 | Recall@300w | Words@5 | MRR | Hit@1 | ms/query |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| section:40/10 | vector | 113 | 0.920 | 0.980 | 120 | 0.872 | 0.820 | 3.0 |
| section:40/10 | keyword | 113 | 0.830 | 0.910 | 113 | 0.650 | 0.500 | 2.0 |
| section:40/10 | hybrid | 113 | 0.910 | 0.980 | 125 | 0.757 | 0.620 | 4.9 |
| section:80/15 | vector | 106 | 0.930 | 0.960 | 130 | 0.871 | 0.820 | 2.8 |
| section:80/15 | keyword | 106 | 0.830 | 0.910 | 129 | 0.633 | 0.480 | 1.9 |
| section:80/15 | hybrid | 106 | 0.890 | 0.960 | 139 | 0.753 | 0.620 | 5.4 |
| section:160/30 | vector | 104 | 0.950 | 0.980 | 141 | 0.876 | 0.820 | 2.9 |
| section:160/30 | keyword | 104 | 0.830 | 0.910 | 160 | 0.633 | 0.480 | 1.8 |
| section:160/30 | hybrid | 104 | 0.910 | 0.960 | 164 | 0.747 | 0.600 | 4.9 |
| fixed:40/10 | vector | 99 | 0.950 | 0.970 | 180 | 0.872 | 0.820 | 2.8 |
| fixed:40/10 | keyword | 99 | 0.830 | 0.890 | 167 | 0.689 | 0.580 | 1.8 |
| fixed:40/10 | hybrid | 99 | 0.920 | 0.980 | 182 | 0.766 | 0.640 | 5.0 |
| fixed:80/15 | vector | 50 | 0.970 | 0.970 | 315 | 0.906 | 0.860 | 3.3 |
| fixed:80/15 | keyword | 50 | 0.900 | 0.900 | 281 | 0.752 | 0.640 | 1.7 |
| fixed:80/15 | hybrid | 50 | 0.920 | 0.940 | 323 | 0.827 | 0.740 | 4.7 |
| fixed:160/30 | vector | 37 | 0.990 | 0.990 | 411 | 0.930 | 0.860 | 3.1 |
| fixed:160/30 | keyword | 37 | 0.920 | 0.920 | 346 | 0.807 | 0.700 | 1.8 |
| fixed:160/30 | hybrid | 37 | 0.960 | 0.940 | 422 | 0.875 | 0.800 | 4.9 |

## Can similarity alone refuse unanswerable questions?

The best balanced-accuracy cut-off on top vector similarity, per chunker (similarity does not depend on the search mode). If the answerable share kept is well below 1, a similarity gate cannot be the refusal mechanism.

| Chunker | Threshold | Answerable kept | Unanswerable refused | Lowest answerable |
| --- | ---: | ---: | ---: | ---: |
| section:40/10 | 0.434 | 0.78 | 1.0 | 0.14 |
| section:80/15 | 0.434 | 0.78 | 1.0 | 0.14 |
| section:160/30 | 0.434 | 0.78 | 1.0 | 0.14 |
| fixed:40/10 | 0.406 | 0.76 | 1.0 | 0.13 |
| fixed:80/15 | 0.396 | 0.78 | 0.9 | 0.12 |
| fixed:160/30 | 0.396 | 0.76 | 0.9 | 0.12 |

## Misses (answerable questions with recall below 1)

- section:40/10 vector: q03, q05, q18, q25, q42
- section:40/10 keyword: q03, q05, q06, q14, q18, q24, q25, q53, q55
- section:40/10 hybrid: q03, q05, q18, q25, q53
- section:80/15 vector: q03, q05, q18, q25
- section:80/15 keyword: q03, q05, q06, q14, q18, q24, q25, q53, q55
- section:80/15 hybrid: q03, q05, q06, q18, q25, q53
- section:160/30 vector: q03, q05, q18
- section:160/30 keyword: q03, q05, q06, q14, q18, q24, q25, q53, q55
- section:160/30 hybrid: q03, q05, q06, q18, q53
- fixed:40/10 vector: q05, q25, q42
- fixed:40/10 keyword: q03, q05, q06, q10, q18, q25, q52, q53, q55
- fixed:40/10 hybrid: q05, q06, q25, q42, q53
- fixed:80/15 vector: q25, q42
- fixed:80/15 keyword: q03, q18, q25, q52, q53
- fixed:80/15 hybrid: q03, q18, q25, q53
- fixed:160/30 vector: q42
- fixed:160/30 keyword: q03, q18, q25, q53
- fixed:160/30 hybrid: q18, q53
