# KernelSight source provenance and license verification

Audit date: 18 November 2026 (Asia/Muscat). Conclusion on source identity:
**VERIFIED MATCH**, not a partial match. Publication/third-party notice completeness
is a separate question and remains unresolved. No connector, model or release was
built or published; no raw file was overwritten.

## 1. Repository and revision evidence

Official repository: [williamhtan/kernelsight](https://huggingface.co/datasets/williamhtan/kernelsight).
Candidate and now recommended immutable SOURCE_REVISION:

```text
d8cd8e804e5b619aadbb082a1fb3a8791b60a680
```

The public [revision API](https://huggingface.co/api/datasets/williamhtan/kernelsight/revision/d8cd8e804e5b619aadbb082a1fb3a8791b60a680)
returned that exact SHA and repository identity. The current repository API also
reported it as HEAD at audit time; the connector must still pin the SHA, not main.
Repository metadata reported public, not gated, and lastModified
2026-06-09T01:34:13Z (an upstream event timestamp, not our audit time).
The [commit history endpoint](https://huggingface.co/api/datasets/williamhtan/kernelsight/commits/main)
returned 19 entries, ending at an initial commit. The candidate's title concerns
adding wgmma files. Earlier entries include other motif uploads, README, and a
change storing small NPZs as ordinary Git blobs. Commit-title “snapshot” counts
are not trace counts: use actual file sets and official split lists.

The pinned README names KernelSight v4; MANIFEST_v4.md names v4.0, 1,444 snapshots,
24x512 inputs, 12/73 vocabularies, and official counts. The card and manifest, not
just the latest-commit title, establish which dataset release this snapshot is.
Local `.cache/huggingface/download/<path>.metadata` contains the same revision
for **all 4,345 files**. The cached revision tree agrees with the live pinned tree
for every file's blob ID and size. Cache evidence corroborates the live check;
a revision-named local cache alone would not prove origin.

## 2. Complete file comparison

Evidence directory: `outputs/reports/phase2_source_audit/` (local, not published).
It contains fetched revision/current metadata, commit history, the complete
paginated tree, per-file integrity records, summary, pinned source documentation,
current third-party license texts, and a repeatable audit script.

| Category | Upstream files | Matched local files |
|---|---:|---:|
| Input tensor NPZs | 1,444 | 1,444 |
| Label NPZs | 1,444 | 1,444 |
| Fingerprint NPZs | 1,444 | 1,444 |
| All official/analysis split JSONs | 7 | 7 |
| Original README/schema/manifest | 3 | 3 |
| Original tools | 2 | 2 |
| .gitattributes | 1 | 1 |
| **Total** | **4,345** | **4,345** |

Verified local size totals **60,902,447 bytes**. Missing files: zero. Mismatches:
zero. Extra files inside the relocated raw dataset tree: zero. Root-level research
README, scripts, reports and visualizations are not upstream files and are not
claimed to belong to the source snapshot.

Relocation mapping was explicit: upstream kernels/splits/tools map under
`data/kernelsight_v4/`; original README.md, data_info.md and MANIFEST_v4.md map to
`docs/dataset/`; .gitattributes remains at project root. Root compatibility symlinks
are not additional upstream data. The three small documentation files were fetched
again at the pinned revision; **no tensor files were downloaded again**.

The pinned tree was retrieved in 11 API pages. Per-file checking used:

- **4,241 ordinary Git files:** size and Git blob SHA-1 computed as
  `SHA1(b'blob '+decimal_length+b'\0'+content)` match upstream tree `oid`.
- **104 Git LFS payloads:** local content SHA-256 and size match the upstream
  `lfs.oid` and payload size. The Git blob OID here describes the LFS pointer,
  not downloaded NPZ content, and was not incorrectly compared to payload bytes.
- A local SHA-256 was additionally recorded for every file. For ordinary Git
  files the upstream comparison is Git SHA-1, **not** an independently supplied
  upstream SHA-256. This provides strong reproducibility evidence, not a claim
  that Git SHA-1 is collision-proof against an adversary.

Hugging Face ETags can denote a regular Git blob ID or an LFS content hash;
quotes, weak prefixes or transport headers are not new hash algorithms. Local
metadata ETags were retained as corroboration, not treated uniformly as SHA-256.
The branch revision is a commit ID, not a file content checksum. Xet/transport
storage does not change the distinction between tree blob and LFS payload IDs.

Official split path union equals the entire pinned set of 1,444 input paths;
counts are 1,124/160/160 and paths are pairwise disjoint. Split contents were read
as JSON; held-out NPZs were **hashed as opaque bytes**, never decoded for design.

| Official split | Local SHA-256 (also source content after Git verification) |
|---|---|
| train.json | `f34177dcc10cb87cee541522e3a0b90c3cf52c56b5c908f8d1e490d6ebd722cd` |
| val.json | `b2fcfad1baef28d33c4d051eb41f2eb56afd7061e6a48d21bd7c94cdc166a67b` |
| test.json | `ef6e57bf6be40161c602d0933ed1efa1ccd84c83529e1b9c766b0b3bc2c575d0` |

The recommendation to pin the candidate is now supported by the entire snapshot,
not merely one example or a download log. This does not verify collector accuracy,
true instruction-level concurrency, or model generalization.

## 3. How to reproduce the source check

From the project root, with network access approved by the execution environment:

```bash
python3 outputs/reports/phase2_source_audit/audit_source.py
```

This standard-library audit invokes curl with TLS verification enabled, downloads
only metadata/file trees and three small documents, and hashes local bytes. It
writes audit evidence, not raw data. It maps relocated files explicitly. The core
script regenerates file_integrity.json and base summary; the richer accompanying
summary additionally reports category counts, cache-tree agreement, split union
and extra-file checks. Those are direct counts/set comparisons over its output,
the cache tree and official JSON lists, not additional tensor decoding.

The browser service could not retrieve HF endpoints; curl successfully did.
The first Python urllib attempt lacked a usable certificate chain; the successful
retry used curl's verified TLS path, without disabling certificate validation.
This network difference is recorded so a failed browser lookup is not mistaken
for evidence that the revision is missing.

## 4. License declaration and third-party evidence

**Confirmed:** the hash-matched [pinned README](https://huggingface.co/datasets/williamhtan/kernelsight/blob/d8cd8e804e5b619aadbb082a1fb3a8791b60a680/README.md)
declares Apache-2.0 both in YAML and in its license/provenance prose. This is more
than a badge. It attributes KernelBench to MIT and CUTLASS to BSD-3-Clause, and
states the distribution contains derived aggregated counter tensors/labels rather
than third-party source code. It provides a William Tan 2026 citation. The manifest
identifies CUTLASS 3.x Hopper examples but does not pin collector/workload commits.

**Confirmed omission:** no LICENSE, NOTICE or COPYING-named file exists anywhere
in the 4,345-file pinned tree. Thus no bundled full license or complete third-party
notice set could be verified. This does not negate the explicit README declaration;
it leaves notice completeness and source-version lineage unresolved.

Primary license references inspected:

- [Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0), especially §§2 and 4:
  permits conversion/distribution subject to conditions, including license delivery,
  change notices and preservation of applicable notices. A source NOTICE, if any,
  has additional retention requirements. No source NOTICE was found here.
- [KernelBench official LICENSE](https://github.com/ScalingIntelligence/KernelBench/blob/main/LICENSE):
  MIT; copyright identifies Anne Ouyang, Simon Guo, Azalia Mirhoseini and the Scaling
  Intelligence Lab. Its notice/permission text must accompany covered copies.
- [CUTLASS official LICENSE.txt](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt):
  BSD-3-Clause, NVIDIA copyright; notice/disclaimer preservation and non-endorsement.
  **The inspected current file also names a separately licensed python/CuTeDSL
  subtree.** Do not claim all current CUTLASS contents are BSD-3-Clause.

The two current official files are saved with `_current` names in audit evidence.
They corroborate general project licensing but are **not** proof of the exact
license texts applicable to KernelSight's collection-time dependency commits.
No evidence here establishes that the CuTeDSL exception applies to the Hopper
examples used in this dataset; nor can we exclude every special-file license
without exact collection revisions. Do not retroactively apply main-branch notices
or invent a source-version legal chain.

## 5. Interpretation, obligations and publication gate

**Interpretation, conditional on the README grant and applicable third-party
rights:** TimeF conversion is contemplated by the stated permissive dataset terms;
format conversion alone is not a new restriction. The available evidence does
not justify asserting that every notice obligation has already been satisfied.
Whether the derived measurements include protectable third-party material is not
established by a hash check or the dataset author's short summary.

A connector written independently with synthetic fixtures and **no dataset bytes**
is different from distributing converted data. It can follow TimeNet's MIT
contribution terms, retain attribution/source links, and fetch from the user-pinned
source at build time. Copying upstream loader/source code into the connector
would need its own license review. Synthetic fixtures must actually be synthetic,
not renamed or lightly edited excerpts. No namespace approval is implied by a
permissive license.

Before publishing a compiled release, obtain or confirm:

1. Full Apache-2.0 license delivery with the artifact; preserved source card,
   provenance/citation and a clear conversion/change notice.
2. Applicable third-party copyright/license/disclaimer texts and collection-time
   workload/collector commit references, or authoritative clarification of what
   material/notices the dataset release requires recipients to retain.
3. A review of any special-file licenses in the actual used dependencies; do not
   assume the current CuTeDSL exception is either relevant or irrelevant by name.
4. A release-packaging mechanism that actually distributes these notices. A
   dataset.yaml license string alone does not deliver license text. Check whether
   TimeNet's ordinary release artifacts support the chosen notice packaging; if
   not, agree an accompanying published license bundle/documentation mechanism.
5. Source author/maintainer clarification if completeness remains uncertain;
   publication should remain withheld until the exact missing evidence is closed.

Proposed card: dataset license Apache-2.0, exact source URL/revision, source release
v4 in description, compiled dataset_version 1.0.0, author citation, original
partitions, and third-party provenance in documentation/attribution. Do not list
MIT merely because the connector implementation is MIT, or claim TimeNet owns
the source dataset. Preserve the distinction among code, data and third-party terms.

**Conclusion:** source bytes and immutable revision are fully verified. Apache-2.0
is explicitly confirmed as the dataset declaration. Complete redistribution notice
and collection-version evidence is **not** verified. Local connector development
can proceed within the research boundary; do not publish compiled data yet.
