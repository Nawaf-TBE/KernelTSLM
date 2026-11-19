# Publication manifest and current blockers

Target: Nawaf-TBE/kernel-xray, private. Description: Explaining GPU execution from hardware time series.

The project root is being initialized independently on main for private publication.
The separate TimeNet worktree and its history remain untouched. GitHub CLI is
authenticated as Nawaf-TBE using an ignored project-local configuration directory.

Candidates: project-authored src/, scripts/, tests/, configs/, README, AGENTS,
CONTRIBUTING, requirements, pyproject, method documents and results/cnn/.
The historical README is preserved in docs/README-before-collaboration.md.

Local only: data/, compatibility aliases, outputs/, environments, caches,
credentials, .worktrees/, checkpoints, and verbatim docs/dataset/ pending license
review. No dataset content or official splits are included in results/cnn/.
Results are copied/derived from verified aggregate metrics and historical plots;
provenance.json records originals and SHA256 checksums.

Before publishing, authenticate gh, inspect any existing target repository and
its history, review an explicit staged allowlist for secrets/large/third-party
files, and verify private visibility. Do not overwrite an unrelated remote.
No teammate invitation is authorized without a username. No checkpoint Release
asset is proposed: redistribution permissions remain unresolved. Authorized
collaborators should arrange approved private transfer with the owner.

Historical audit references into outputs/ are local evidence links, unavailable
in a source-only clone; the copied aggregate verdict and results are available
under results/cnn/. A self-contained fresh-machine TimeNet setup remains pending
publication/namespace and third-party licensing review.
