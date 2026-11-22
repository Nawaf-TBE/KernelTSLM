# Diagnosis: what limits a kernel ("Talk to Your GPU")

**Question.** What limits this kernel, and what should I try? Free text, `options: []`, all 24 Channels with `SMs Active` last (it is > 0 exactly in busy Bins, so the answer can read the busy timeline). Answer = up to 3 fact sentences, then `Limiter: <limiter>.` and `Try: <step>`.

**Rules** (busy Bins only; first match wins; percentiles vs the 862 training Snapshots only):
1. launch gaps: idle ≥ 50% of the active span (first to last busy Bin) and ≥ 10 busy periods.
2. low SM utilization: SMs Active < 50%.
3. busiest unit ≥ 25% of peak: DRAM traffic (read + write) → memory bandwidth; Tensor Active → tensor cores; pipe: fma / alu / xu → FMA / ALU / XU pipe.
4. barrier stalls: stall: barrier in the top 10% of training.
5. latency: otherwise.

**Try** (fixed table, 12 steps, `score.TRIES`): one step per limiter. Three limiters have a condition. Memory → tiling if Tensor Active or FMA ≥ 10%, else fewer bytes. FMA → bf16/TF32 if Tensor Active < 5%. Latency → coalesce if gmem_coalesce_ratio is in the bottom 25%.

**Facts** (checked by `score.py`): `<unit> is N% of peak while busy.` (±10 points); `The GPU is idle N% of the active span, across K busy periods.`; `<Channel> is in the top|bottom k% of training while busy.`

**Sanity (all 1,444 Snapshots).**
- activation 82/91 memory; softmax 14/14; pooling 35/42 ALU.
- CUTLASS fp16/bf16 GEMM 210/278 tensor cores; ws_overlap 325/472.
- KernelBench fp32 matmul 38/54 FMA pipe → "Use bf16 or TF32". These Problems have no dtype Variants, so we cannot measure that step.
- memory_movement: 40/48 low SM utilization, NOT memory-bound (DRAM ≈ 1%, SMs Active 5-18%).
- dtype pairs (92 pairs, 46 non-matmul Problems, fp32 vs bf16, busy time = busy Bins × Bin width): memory-bound fp32 → bf16 is 1.68× faster (median, n 68, IQR 1.17-1.87). ALU-bound: 1.01× (n 18). No pair flips to tensor cores. The memory limiter predicts a real speedup.

**Rows.** train 862, validation 291, test 291. Test limiters: tensor 117, memory 65, low SM 52, latency 25, launch gaps 15, FMA 9, ALU 6, barrier 2. Gemma 3 tokens: text lines + prompt 820; answer mean 53, max 82; `max_new_tokens` 90.

**Controls.** shuffled = rules on the shuffled Bins: busy means stay, the timeline changes. The answer changes in 268/291 test rows (mostly to launch gaps). reversed = identical answer in 1,444/1,444 Snapshots (robustness only; skip it on GPU).

**Metric.** main = limiter accuracy (no valid `Limiter:` = wrong). Also balanced accuracy, Try accuracy, claim precision, unparsable share, per class. Self-test 1.0; random .110 (chance 1/9).

**Baselines (test; limiter acc / balanced / Try).**
| | original | shuffled |
|---|---|---|
| majority (tensor cores) | .402 / .125 / .402 | |
| true workload class → train majority | .570 / .248 / .557 | |
| MiniRocket class → train majority | .574 / .250 / .546 | |
| LR on text-line mean/std (C on val) | **.938** / .819 / .938 | .402 |
| original reference, order-blind | 1.0 | .399 (claims .846) |

**GPU time.** Training 862 × 0.38 s = 5.5 min per epoch. Generation ≈ 0.11 s prompt + 54 × 0.052 s = 2.9 s per row. Validation 14 min. Test real + shuffled 28 min. Full ≈ 5.5 min/epoch + 42 min. Round-1 minimum: 100 validation rows + test real + shuffled ≈ 33 min + training.

**Risks.**
- The labels come from rules, not from expert annotation. They are heuristics, not ground truth.
- The text line nearly gives the answer: LR reaches .938. Order use shows only on shuffled input.
- Bins are coarse (median 642 µs), so busy time is approximate.
- Few FMA, ALU and barrier rows in test (9, 6, 2).
- One run.

**Samples** (prompt elided):
```
FMA/matmul L1_P10: "pipe: fma is 55% of peak while busy. The GPU is idle 50% of the active span, across 33 busy periods. Tensor Active is 0% of peak while busy. Limiter: FMA pipe. Try: Use bf16 or TF32 so the tensor cores run."
memory/activation L1_P20 bf16: "DRAM traffic is 70% of peak while busy. The GPU is idle 35% of the active span, across 33 busy periods. L2 hit rate is 40% while busy. Limiter: memory bandwidth. Try: Move fewer bytes: fuse with neighbouring ops or use a narrower data type."
low SM/scatter N256K_B16: "SMs Active is 5% of peak while busy. gmem_coalesce_ratio is in the bottom 5% of training while busy. Limiter: low SM utilization. Try: Use more threads and blocks (a bigger grid or split-K) to fill all SMs."
```
