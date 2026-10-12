# Per-layer routing entropy on the recorded coding-agent traces

Research note for wayfinder ticket [#17](https://github.com/kido5217/Infernix/issues/17). Answers one
question: **does per-layer routing entropy vary enough on our own traces to support an entropy-derived
per-layer prior at all?** Measurements read from the eight version-2 `NRTRACE1` route traces under
`out/route-traces/` (recorded by ticket [#11](https://github.com/kido5217/Infernix/issues/11)),
run 2026-10-13 on `main` @ `4cc480fa`.

## Verdict

**Not flat — a static per-layer statistic has real, stable structure here.** Per-layer routing
entropy spans **1.50 bits** pooled across the eight sessions (6.981 at layer 15 to 8.482 at layer 0;
1.73 bits across the 48 layer means) — about a fifth of the `log2(512)=9`-bit maximum — and the layer
ordering is reproducible rather than noise:

- the per-layer spread across the eight sessions is only **0.155 bits** on average (within one regime,
  0.079 bits long / 0.113 bits short), so the across-layer **layer-mean** range (1.73 bits; 1.50 bits
  pooled) is **11x** (pooled: **9.7x**) the session-to-session variation **for the same layer**;
- a within-session bootstrap noise floor is **0.023 bits at each session's own length** and
  **0.036 bits at the 400-token control budget** — **74x** and **49x** smaller than the layer-mean
  range (**42x** against the pooled range at the control budget);
- first-half vs second-half **test-retest Spearman is 0.94** (min 0.89), and cross-session
  **pairwise Spearman averages 0.83** (min 0.57) — the layers keep their rank across all eight
  sessions, and the long- and short-regime orderings are related at Spearman 0.74 — a systematic
  regime shift, not the near-identity seen within a regime (0.93 long / 0.95 short).

So the entropy premise is **not** dead on arrival; the earlier expectation that it would be flat is
falsified by the data. **What this does not establish** is that an entropy-derived prior *improves
the cache*: entropy is a near-perfect proxy for concentration here (Spearman &#8722;0.99 against the
pooled top-64 share), which is a static popularity statistic — not the hit-rate-response curve
MoE-CORE allocates on — and this repo's own replays already lost 3-34% to *hard* per-layer quotas
(`docs/maintainer/qwen3_8-flash-next-design.md` §9.3, rows for "LRU, equal per-layer quotas";
the [MoE-CORE
note](https://github.com/kido5217/Infernix/blob/research/moe-core-layer-allocator/docs/research/moe-core-layer-allocator.md)
records the paper's own +1.1% ablation of non-uniform capacity). The cache-response test that would
settle the arm remains open.

## Data and convention

Eight greedy-decoding coding-agent sessions from `bench/route_traces/manifest.json`, written by
`--route-trace`: four long (prompt >= 32K tokens) and four short (<= 2K tokens, 3-5 turns).
Generation rounds only (`Decode` + `Verify` + `ForcedTokens`; `PrefillChunk` is a separate regime
and is excluded). One routed use = one **(live generation column, top-k slot)**; the live mask is
exactly the accepted generated tokens, so the count of live generation columns equals the manifest's
`generated_tokens` for every session (4504 / 1713 / 1126 / 1567 / 566 / 467 / 632 / 614) — used in
the script as a reader cross-check. There are no `ForcedTokens` rounds in these traces; per layer,
uses = `generated_tokens x top_k` = tokens x 10. No column repeats an expert id.

**Normalization.** A layer's expert-usage distribution is its empirical histogram over those uses:
`p_e = c_e / sum_e c_e`. **Unselected experts carry probability zero and are dropped**, so `H` is the
entropy of the observed distribution, `H = -sum p_e log2 p_e` (0..9 bits), and its support size is the
distinct-active-expert count. Concentration is the share of a layer's uses taken by its 64 most-used
experts. "Temporal drift" splits each session's generation tokens, in order, into equal halves.

## The 48-layer table

`H` is bits; `mean H`/`std` are across the eight sessions; `top-64` and `distinct` are pooled.

| layer | pooled H | long-1 H | long-2 H | long-3 H | long-4 H | short-1 H | short-2 H | short-3 H | short-4 H | mean H | std | pooled top-64 | pooled distinct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8.4822 | 8.3584 | 8.2171 | 8.2112 | 8.2557 | 8.3941 | 8.4035 | 8.4874 | 8.3865 | 8.3392 | 0.0935 | 0.3324 | 497 |
| 1 | 8.2916 | 8.1675 | 7.9266 | 7.8814 | 7.9528 | 8.0793 | 8.0306 | 8.1520 | 8.1971 | 8.0484 | 0.1119 | 0.3848 | 502 |
| 2 | 8.3492 | 8.2004 | 7.9770 | 7.8887 | 7.9944 | 8.2021 | 7.9376 | 8.1436 | 8.1615 | 8.0632 | 0.1187 | 0.3839 | 496 |
| 3 | 7.6973 | 7.5403 | 7.3632 | 7.3608 | 7.3632 | 7.4502 | 6.8953 | 7.3323 | 7.5685 | 7.3592 | 0.1939 | 0.5780 | 498 |
| 4 | 7.9150 | 7.7658 | 7.5078 | 7.5546 | 7.5917 | 7.5302 | 7.2194 | 7.5515 | 7.7744 | 7.5619 | 0.1616 | 0.5094 | 499 |
| 5 | 8.0285 | 7.8046 | 7.7714 | 7.8033 | 7.7848 | 7.6751 | 7.3105 | 7.6810 | 7.7361 | 7.6958 | 0.1532 | 0.4616 | 483 |
| 6 | 8.2618 | 8.0806 | 7.8996 | 7.9091 | 7.9437 | 7.9989 | 7.7368 | 8.0265 | 8.0332 | 7.9535 | 0.1011 | 0.3844 | 491 |
| 7 | 8.2492 | 8.0400 | 7.9108 | 7.9038 | 7.9413 | 8.0306 | 7.8751 | 8.0194 | 8.0698 | 7.9738 | 0.0694 | 0.4000 | 487 |
| 8 | 7.9403 | 7.7218 | 7.4768 | 7.3600 | 7.4710 | 7.7267 | 7.3666 | 7.6583 | 7.6358 | 7.5521 | 0.1419 | 0.4889 | 478 |
| 9 | 7.8845 | 7.6108 | 7.4589 | 7.3304 | 7.4493 | 7.6482 | 7.2703 | 7.5915 | 7.6260 | 7.4982 | 0.1343 | 0.4980 | 472 |
| 10 | 8.0758 | 7.8217 | 7.5184 | 7.4287 | 7.5302 | 8.0567 | 7.8257 | 7.9704 | 8.0116 | 7.7704 | 0.2302 | 0.4576 | 488 |
| 11 | 7.8543 | 7.5485 | 7.3209 | 7.1478 | 7.3061 | 7.8691 | 7.3739 | 7.7388 | 7.7830 | 7.5110 | 0.2461 | 0.5248 | 481 |
| 12 | 8.1698 | 7.9375 | 7.7177 | 7.7273 | 7.7690 | 8.0745 | 7.9030 | 8.0039 | 8.0734 | 7.9008 | 0.1383 | 0.4095 | 488 |
| 13 | 8.1682 | 7.8272 | 7.6948 | 7.5987 | 7.7117 | 8.1365 | 8.0572 | 8.0865 | 8.0947 | 7.9009 | 0.2022 | 0.4343 | 502 |
| 14 | 7.8446 | 7.5489 | 7.2440 | 7.0609 | 7.1998 | 7.9648 | 7.6739 | 7.8669 | 7.8995 | 7.5573 | 0.3288 | 0.5322 | 469 |
| 15 | 6.9809 | 6.5609 | 6.4705 | 6.3290 | 6.3843 | 7.1568 | 6.4405 | 6.8708 | 7.2568 | 6.6837 | 0.3396 | 0.7529 | 456 |
| 16 | 7.3897 | 7.0037 | 6.8315 | 6.7473 | 6.8392 | 7.2925 | 6.8821 | 7.2068 | 7.4635 | 7.0333 | 0.2414 | 0.6436 | 457 |
| 17 | 7.8183 | 7.4919 | 7.3286 | 7.2095 | 7.2898 | 7.8952 | 7.6325 | 7.6229 | 7.9143 | 7.5481 | 0.2500 | 0.5257 | 475 |
| 18 | 8.1739 | 7.8869 | 7.7183 | 7.6957 | 7.7640 | 8.1400 | 7.9916 | 8.0878 | 8.1275 | 7.9265 | 0.1735 | 0.4263 | 499 |
| 19 | 7.7830 | 7.4139 | 7.2723 | 7.1413 | 7.2265 | 7.8653 | 7.5976 | 7.8033 | 7.9358 | 7.5320 | 0.2914 | 0.5314 | 475 |
| 20 | 7.7870 | 7.4705 | 7.3633 | 7.2731 | 7.3729 | 7.6541 | 7.3913 | 7.7292 | 7.7555 | 7.5012 | 0.1734 | 0.5378 | 479 |
| 21 | 8.1726 | 7.8996 | 7.7988 | 7.7651 | 7.8044 | 7.9434 | 7.8014 | 7.9593 | 7.8926 | 7.8581 | 0.0696 | 0.4267 | 506 |
| 22 | 8.2889 | 8.0941 | 7.9926 | 7.9883 | 8.0375 | 8.0886 | 7.9436 | 8.1572 | 8.0349 | 8.0421 | 0.0644 | 0.3777 | 508 |
| 23 | 7.6658 | 7.3186 | 7.2698 | 7.1979 | 7.2821 | 7.5074 | 7.1107 | 7.5478 | 7.6628 | 7.3621 | 0.1778 | 0.5914 | 503 |
| 24 | 7.8131 | 7.5336 | 7.3663 | 7.2642 | 7.3311 | 7.7188 | 7.4603 | 7.7145 | 7.7130 | 7.5127 | 0.1742 | 0.5239 | 478 |
| 25 | 7.7255 | 7.4454 | 7.2940 | 7.2135 | 7.2114 | 7.4816 | 7.1619 | 7.4783 | 7.5647 | 7.3564 | 0.1436 | 0.5419 | 464 |
| 26 | 7.9490 | 7.6549 | 7.4659 | 7.3412 | 7.3943 | 7.9722 | 7.8720 | 7.8409 | 7.9894 | 7.6914 | 0.2464 | 0.4891 | 481 |
| 27 | 7.7446 | 7.4670 | 7.2788 | 7.1105 | 7.1723 | 7.7610 | 7.4320 | 7.6377 | 7.6804 | 7.4425 | 0.2255 | 0.5431 | 472 |
| 28 | 8.0356 | 7.8084 | 7.6248 | 7.5829 | 7.6271 | 7.8095 | 7.6375 | 7.7085 | 7.7755 | 7.6968 | 0.0852 | 0.4741 | 485 |
| 29 | 8.0483 | 7.8049 | 7.6821 | 7.5284 | 7.6046 | 7.9391 | 7.9194 | 7.9449 | 7.9022 | 7.7907 | 0.1542 | 0.4665 | 501 |
| 30 | 7.8130 | 7.5874 | 7.3450 | 7.1396 | 7.2590 | 7.7909 | 7.5565 | 7.6916 | 7.7206 | 7.5113 | 0.2213 | 0.5293 | 466 |
| 31 | 6.9898 | 6.7086 | 6.6381 | 6.4930 | 6.4834 | 6.8617 | 6.3105 | 6.5954 | 6.7865 | 6.6096 | 0.1676 | 0.7452 | 452 |
| 32 | 7.3885 | 7.1000 | 6.9595 | 6.7862 | 6.8454 | 7.1670 | 6.8243 | 7.0789 | 7.1706 | 6.9915 | 0.1478 | 0.6445 | 460 |
| 33 | 7.8062 | 7.5558 | 7.3059 | 7.1601 | 7.2776 | 7.7797 | 7.6264 | 7.5961 | 7.6863 | 7.4985 | 0.2073 | 0.5267 | 467 |
| 34 | 8.1397 | 7.9042 | 7.7960 | 7.7587 | 7.7857 | 7.9484 | 7.9023 | 7.8951 | 7.8750 | 7.8582 | 0.0641 | 0.4298 | 495 |
| 35 | 7.8164 | 7.5449 | 7.4190 | 7.3444 | 7.3820 | 7.7002 | 7.5171 | 7.6434 | 7.6156 | 7.5208 | 0.1212 | 0.5354 | 488 |
| 36 | 7.6060 | 7.3551 | 7.3141 | 7.2779 | 7.2774 | 7.3764 | 7.1223 | 7.4252 | 7.3548 | 7.3129 | 0.0859 | 0.5841 | 452 |
| 37 | 8.0074 | 7.7709 | 7.6825 | 7.6709 | 7.6957 | 7.6871 | 7.6229 | 7.7913 | 7.6548 | 7.6970 | 0.0532 | 0.4692 | 472 |
| 38 | 8.1990 | 8.0440 | 7.9546 | 7.9495 | 8.0126 | 7.7852 | 7.7789 | 7.9415 | 7.8292 | 7.9119 | 0.0951 | 0.4067 | 492 |
| 39 | 7.3896 | 7.1067 | 7.1252 | 7.1575 | 7.1021 | 7.1085 | 6.7647 | 7.0749 | 7.1410 | 7.0726 | 0.1187 | 0.6606 | 477 |
| 40 | 7.2881 | 7.0727 | 7.0714 | 7.1102 | 7.0702 | 6.9946 | 6.5478 | 6.9388 | 6.9485 | 6.9693 | 0.1699 | 0.6804 | 488 |
| 41 | 7.6208 | 7.4010 | 7.3852 | 7.3863 | 7.4280 | 7.3363 | 7.0398 | 7.2825 | 7.2681 | 7.3159 | 0.1171 | 0.5869 | 479 |
| 42 | 7.8332 | 7.5847 | 7.6242 | 7.6928 | 7.6531 | 7.6198 | 7.5197 | 7.5931 | 7.5427 | 7.6038 | 0.0528 | 0.5142 | 476 |
| 43 | 7.4637 | 7.2447 | 7.3226 | 7.3470 | 7.2787 | 7.1008 | 6.8721 | 7.0604 | 7.2047 | 7.1789 | 0.1492 | 0.6358 | 488 |
| 44 | 7.4712 | 7.2818 | 7.2510 | 7.3070 | 7.3058 | 6.9797 | 7.0339 | 7.1744 | 7.0859 | 7.1775 | 0.1213 | 0.6162 | 479 |
| 45 | 7.5453 | 7.3487 | 7.3363 | 7.3260 | 7.3983 | 7.2372 | 7.0936 | 7.3934 | 7.2203 | 7.2942 | 0.0971 | 0.6024 | 474 |
| 46 | 7.6122 | 7.4750 | 7.4313 | 7.4158 | 7.4705 | 7.1959 | 7.0819 | 7.2624 | 7.2888 | 7.3277 | 0.1342 | 0.5873 | 485 |
| 47 | 7.7509 | 7.5399 | 7.5829 | 7.6113 | 7.6366 | 7.4726 | 7.3990 | 7.5636 | 7.4975 | 7.5379 | 0.0733 | 0.5405 | 483 |

## Stability — the comparison that decides the ticket

Across-layer range of entropy vs the spread for the same layer across sessions, and the rank
correlation of layers across sessions (raw script lines):

```
across-layer entropy range (full data)   per session: long-1=6.561..8.358, long-2=6.470..8.217, long-3=6.329..8.211, long-4=6.384..8.256, short-1=6.862..8.394, short-2=6.311..8.403, short-3=6.595..8.487, short-4=6.786..8.387
across-layer range of the POOLED entropy: 6.981..8.482 (R=1.5013 bits)
per-layer entropy spread across 8 sessions: mean std=0.1548 bits, median=0.1457, max=0.3396 bits
  within regime: long range=1.8244, long across-session std=0.0788 | short range=1.7794, short across-session std=0.1134
  long-group mean vs short-group mean layer ordering: Spearman=0.751
sample-size-controlled (T=400): across-layer range=6.577..8.295 (R=1.7180), per-layer across-session std mean=0.1546
within-session bootstrap noise (native T): per-layer std mean=0.0235, max=0.0574 bits
within-session bootstrap noise (T=400): per-layer std mean=0.0356, max=0.0638 bits
pairwise Spearman of layer entropy across sessions: mean=0.827, min=0.574, max=0.982 (of 28 pairs)
split-half test-retest Spearman (first vs second half, per session): mean=0.943, min=0.891, max=0.989
entropy vs pooled top-64 share: Spearman=-0.992
entropy ranges: pooled=1.5013 bits, layer-mean=1.7296 bits
entropy range / per-layer across-session std (layer-mean range) = 11.17x (pooled range: 9.70x)
entropy range / bootstrap noise (native T, layer-mean range) = 73.74x
entropy range / bootstrap noise (T=400, layer-mean range) = 48.61x
entropy range / bootstrap noise (T=400, pooled range) = 42.19x
```

Grouped JSON summary:

```json
{
 "pooled_range": 1.5012970152019083,
 "layer_mean_range": 1.7295825997347993,
 "layer_std_mean": 0.15484028009514908,
 "layer_std_max": 0.33956066202107354,
 "long_range": 1.8244094187867468,
 "long_across_session_std": 0.07877348353630034,
 "short_range": 1.7793561733216565,
 "short_across_session_std": 0.11337696711458246,
 "long_short_ordering_spearman": 0.7505427702996093,
 "subsample_range": 1.7180170604634633,
 "subsample_session_std_mean": 0.15459603917146483,
 "bootstrap_noise_mean": 0.023456584696935706,
 "bootstrap_noise_max": 0.05736581978274898,
 "bootstrap_noise_matched_mean": 0.035580916845633435,
 "bootstrap_noise_matched_max": 0.06382535450716957,
 "noise_budget_tokens": 400,
 "spearman_mean": 0.8266585509583774,
 "spearman_min": 0.573816760746852,
 "spearman_max": 0.9824142422926617,
 "split_half_spearman_mean": 0.9433754884932697,
 "entropy_top64_spearman": -0.9918584455058619,
 "drift_mean_dH": 0.11895745831426263,
 "drift_max_dH": 0.696254540243455,
 "drift_mean_jaccard": 0.773970372223819,
 "drift_mean_novel": 0.027629277946245572
}
```

Reading: the across-layer range (1.50 pooled / 1.73 across layer means / 1.82 within the long regime
/ 1.78 within the short regime) dwarfs both the per-layer across-session std and the bootstrap noise
floor; the pattern survives sample-size control (all sessions subsampled to 400 tokens still span
1.72 bits, and the matched noise floor at that budget is 0.036 bits). It is not a long-vs-short
artifact in the sense of being absent in one regime, but the two regimes do reorder the layers:
their orderings are related at Spearman 0.74 against 0.93 (long) and 0.95 (short) within a regime.
The lowest-entropy (most concentrated) layers are **15, 31, 40, 32, 39, 16** (top-64 share
0.64-0.75); the highest-entropy (most spread) layers are **0, 2, 1, 22, 6, 7, 38, 18** (top-64 share
0.33-0.43). The session-to-session spread is heteroscedastic: the most stable layers are **42, 37, 34,
22, 7, 21** (std 0.05-0.07 bits) while the least stable are **15, 14, 19, 17, 26, 11**
(std 0.25-0.34 bits) — the concentrated low-entropy layers 15 and 14 drift most across sessions, and
layer 0 is the highest-entropy layer and sits in the more stable half (std 0.09 bits, 10th of 48),
though outside the 0.05-0.07 band. The weakest session pair is
long-3 x short-1 at rho = 0.57.

## Temporal drift within a session

First half vs second half of each session, per layer: `dH = |H_first - H_second|`, Jaccard is the
turnover of the used-expert *set* between halves, "novel" is the share of second-half uses by experts
never seen in the first half.

| layer | mean dH | max dH | mean Jaccard | max novel-use share |
|---|---|---|---|---|
| 0 | 0.0784 | 0.1797 | 0.897 | 0.043 |
| 1 | 0.1335 | 0.3957 | 0.846 | 0.053 |
| 2 | 0.1492 | 0.2940 | 0.849 | 0.040 |
| 3 | 0.1620 | 0.2855 | 0.773 | 0.070 |
| 4 | 0.1059 | 0.3216 | 0.801 | 0.053 |
| 5 | 0.1348 | 0.2733 | 0.811 | 0.061 |
| 6 | 0.1431 | 0.3107 | 0.829 | 0.057 |
| 7 | 0.1136 | 0.2548 | 0.833 | 0.042 |
| 8 | 0.0863 | 0.2083 | 0.782 | 0.083 |
| 9 | 0.1004 | 0.1817 | 0.786 | 0.054 |
| 10 | 0.0737 | 0.1375 | 0.803 | 0.060 |
| 11 | 0.1129 | 0.2495 | 0.766 | 0.065 |
| 12 | 0.0660 | 0.1495 | 0.826 | 0.077 |
| 13 | 0.0762 | 0.1792 | 0.800 | 0.059 |
| 14 | 0.1111 | 0.3203 | 0.777 | 0.064 |
| 15 | 0.2414 | 0.6182 | 0.679 | 0.069 |
| 16 | 0.1415 | 0.4202 | 0.707 | 0.074 |
| 17 | 0.1395 | 0.4134 | 0.782 | 0.080 |
| 18 | 0.0582 | 0.1292 | 0.823 | 0.052 |
| 19 | 0.0932 | 0.2598 | 0.777 | 0.059 |
| 20 | 0.0962 | 0.3372 | 0.766 | 0.073 |
| 21 | 0.0638 | 0.1340 | 0.784 | 0.063 |
| 22 | 0.0852 | 0.1870 | 0.816 | 0.041 |
| 23 | 0.1124 | 0.5272 | 0.751 | 0.059 |
| 24 | 0.0873 | 0.3468 | 0.769 | 0.100 |
| 25 | 0.0737 | 0.2570 | 0.748 | 0.053 |
| 26 | 0.0840 | 0.2560 | 0.784 | 0.083 |
| 27 | 0.1002 | 0.3883 | 0.755 | 0.060 |
| 28 | 0.0769 | 0.2152 | 0.797 | 0.054 |
| 29 | 0.0888 | 0.1827 | 0.775 | 0.104 |
| 30 | 0.1074 | 0.3110 | 0.768 | 0.081 |
| 31 | 0.2715 | 0.6963 | 0.665 | 0.138 |
| 32 | 0.1493 | 0.3912 | 0.698 | 0.080 |
| 33 | 0.0950 | 0.2542 | 0.745 | 0.117 |
| 34 | 0.1000 | 0.2408 | 0.805 | 0.055 |
| 35 | 0.0999 | 0.2472 | 0.749 | 0.091 |
| 36 | 0.1012 | 0.2180 | 0.744 | 0.098 |
| 37 | 0.1392 | 0.2980 | 0.784 | 0.063 |
| 38 | 0.1142 | 0.1860 | 0.820 | 0.036 |
| 39 | 0.2036 | 0.6246 | 0.699 | 0.055 |
| 40 | 0.1603 | 0.6159 | 0.682 | 0.051 |
| 41 | 0.1493 | 0.3764 | 0.737 | 0.044 |
| 42 | 0.1023 | 0.3474 | 0.771 | 0.049 |
| 43 | 0.1846 | 0.4003 | 0.727 | 0.035 |
| 44 | 0.1184 | 0.2212 | 0.738 | 0.032 |
| 45 | 0.1362 | 0.2337 | 0.764 | 0.031 |
| 46 | 0.1564 | 0.2543 | 0.767 | 0.032 |
| 47 | 0.1317 | 0.2925 | 0.797 | 0.028 |

drift summary: mean |dH|=0.1190 bits (max 0.6963), mean Jaccard=0.774, mean novel-use share=0.028. So within a session the used-expert set largely persists (mean Jaccard 0.77) and only
~3% of second-half uses are novel; the entropy shift between halves (0.12 bits mean) is smaller than
the across-layer range and comparable to the per-layer across-session spread, i.e. the layer
structure is more stable *across layers* than it is *across time within a layer*.

## What this does and does not establish

- **Establishes:** per-layer routing entropy (and the equivalent concentration and distinct-expert
  statistics) is a large, reproducible per-layer property of these traces — the premise an
  entropy-derived per-layer prior stands on. A single-model, single-engine, coding-agent scope; the
  numbers are of this checkpoint (`Qwen3.8-Flash-Next-Uncensored-NVFP4`) and these eight sessions.
- **Does not establish:** that such a prior helps. Entropy is static popularity, and the repo's
  expert-cache design record plus the MoE-CORE note report that hard per-layer quotas *lost* 3-34% in
  replay and that the exploitable order is temporal (arXiv:2608.18261: LRU 65.9% > static pinning
  59.2% at one budget). An entropy prior would need its own hit-rate-response measurement before any
  adoption.

## Reproduce

From the repository root (host Python 3.13 with numpy, or `nix develop .`):

```
python3 tools/expert_cache_replay/per_layer_routing_stats.py --json out/per-layer-routing-stats.json
```

The script reuses `read_trace` from `tools/expert_cache_replay/host_tier.py` (the engine's canonical
reader) rather than a second parser; it lives under `tools/expert_cache_replay/` because it imports
that reader as a sibling. Seed 17 fixes the bootstrap/subsample draws; `SUBSAMPLE_TOKENS=400`,
`BOOTSTRAP=200`.

## Sources

- Traces: `out/route-traces/{long,short}-{1..4}.bin` (version-2 `NRTRACE1`, 64-byte header, one
  record per round; format `src/models/qwen4_exp/program/route_trace.h`).
- Manifest and protocol: `bench/route_traces/manifest.json`, `out/route-traces/validate.txt`.
- Reader: `tools/expert_cache_replay/host_tier.py` (`read_trace`, `Round`).
- Script: `tools/expert_cache_replay/per_layer_routing_stats.py`.
- Prior art (research branches, not merged): `docs/research/per-layer-telemetry.md` on
  `research/per-layer-telemetry`, and
  [MoE-CORE allocator](https://github.com/kido5217/Infernix/blob/research/moe-core-layer-allocator/docs/research/moe-core-layer-allocator.md)
  on `research/moe-core-layer-allocator`; the 3-34% equal-quota loss is in tree at
  `docs/maintainer/qwen3_8-flash-next-design.md` §9.3. arXiv:2608.18261 (uniform per-layer capacity,
  no entropy), arXiv:2610.01950 (MoE-CORE, no entropy).
