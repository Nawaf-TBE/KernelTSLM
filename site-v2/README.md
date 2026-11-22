# Kernel X-Ray — Hackathon Demo V2

An independent, compact, interactive alternative to the existing `site/` page.
**The existing site and `main` branch are unchanged.**

## Preview

From a local clone:

```bash
git fetch origin
git switch feature/hackathon-demo-v2
open site-v2/index.html
```

Open `site-v2/index.html` in Chrome or Safari. All required local assets are bundled in this folder:

- `index.html` — interactive page and renderer
- `kx.css` — chart-paper visual design
- `demo-data.js` — six recorded KernelSight test traces and original examples
- `audit-records.js` — recorded per-question predictions for the six test traces under evaluated conditions

Do **not** copy only `index.html` into a separate directory.

## What works

1. Select one of six test kernels to see its recorded 24 counter channels (256 display cells per channel summarizing 512 bins) and rule-derived workload/limiter.
2. Select Original, Shuffled, Reversed or Swapped to see a signal transformation and associated **recorded** evaluation outcome.
3. On Original, Shuffled and Reversed, choose among recorded temporal questions for the selected kernel to see the saved model output and relevant reference answer.
4. Swapped shows the recorded classification prediction against the original label, plus the donor's workload where available.
5. Switch between four use cases: diagnosis, temporal QA, segmentation (with actual *predicted and reference* timelines), and classification.
6. Explore compact findings and expand baseline comparisons and research limitations.

**This page replays saved experiment results; it does not perform live model inference.** The visuals for shuffling are illustrative permutations over 256 display samples, while model predictions were collected with a specific fixed shuffle over 512 real bins. For swapping, a representative alternative demo kernel is drawn, not necessarily the exact recorded donor. The displayed numerical prediction and correct answer are always from the experiment record for that condition. Do not confuse these approximations with the inference inputs.

## Verified evidence

- `runs/mt_all/metrics.json` — aggregate scores and trained-model evaluation conditions
- `runs/mt_all/predictions_temporal_{real,shuffled,reversed}.jsonl` — temporal inputs and outputs
- `runs/mt_all/predictions_classification_{real,swapped,shuffled,reversed}.jsonl` — workload predictions and donors
- `runs/mt_all/predictions_segmentation_{real,shuffled,reversed,swapped}.jsonl` — generated timelines
- `runs/mt_all/predictions_diagnosis_{real,shuffled}.jsonl` — diagnosis text
- `runs/usecases/temporal/test.jsonl` — temporal question wording and identifiers
- `usecases/*/SPEC.md` — task definitions, control limitations, and scorer contracts

Metrics from one model seed; diagnostic bottleneck labels are rule-derived; test traces originate from a single H100. See the root README for methods, stronger specialized baselines and further limitations.

## Publication status

On `feature/hackathon-demo-v2`, the original `site/` and `main` remain unchanged. To show judges this version publicly, deploy the complete `site-v2/` folder (not only the HTML file), or create a reviewed PR later.
