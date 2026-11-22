# Kernel X-Ray — Hackathon demo V2

This is an **alternative** to the existing `site/` demo. The original website is unchanged.

## Preview locally

From a clone of the Kernel X-Ray repository:

```sh
git fetch origin
git switch feature/hackathon-demo-v2
open site-v2/index.html
```

Alternatively, open `site-v2/index.html` in Chrome or Safari. The page references `../site/kx.css` and `../site/demo-data.js`, so keep the repository folder structure intact.

## What's different

- Shorter, demo-first introduction with the **order-audit research question**.
- Interactive selector for six recorded test kernels, visualizing 24 counters.
- Original / shuffled / reversed / swapped visualization for one selected channel, with the **published aggregate metrics** for the corresponding audit condition.
- Compact four-capability tabs: diagnosis, temporal QA, timeline segmentation, and workload classification.
- Three brief findings, with links to the full README and original demo.

## Scientific transparency

The real example counter signals and single-kernel predictions originate from the existing `site/demo-data.js`.

The order-audit control transforms a display-scaled sample of one recorded channel in the browser. **It does not execute OpenTSLM**. The displayed metrics are already published *aggregate* scores, **not per-kernel predictions recalculated for the selected visual transformation**. In particular, do not present the displayed model response in the capabilities section as the response to a transformed input.

The timeline visualization in the segmentation tab represents the **reference annotation**, not a generated segmentation for that kernel.

## Before submission

- Visually check desktop and mobile layouts, labels, and contrast.
- If per-example perturbation predictions and their corresponding reference answers are exported, integrate them as an explicit, audited dataset for a genuine paired interactive demonstration.
- Keep the grouped-split evaluation limitations and trained-model benchmarks available in the repository README.

## Scope

V2 is deliberately separate. No edits were made to `site/`, `runs/`, model weights, or the default `main` branch.
