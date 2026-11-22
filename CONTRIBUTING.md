# Contributing

`main` contains reviewed stable work. Create feature branches and merge through
reviewed pull requests; never force-push shared history.

Experiments must record seeds, environment versions, source hashes, official
splits, preprocessing fit partition, selection criteria, metrics, and coverage.
Use training for development and validation for selection; reserve official test
data for final evaluation. Never feed targets or target-bearing metadata into a
counter-only model. Preserve unknown-label masking and report class imbalance.

Preserve raw data, official splits, historical outputs, and frozen checkpoints.
Read AGENTS.md and relevant methods before changes. Run checks appropriate to the
change and report results in the PR. Do not rerun training for documentation edits.

Keep TimeNet changes in its separate worktree. Do not commit environments,
credentials, datasets, compiled views, or checkpoints. Obtain licensing approval
before redistributing third-party materials. Teammate access requires the owner's
explicit invitation after their username is supplied.
