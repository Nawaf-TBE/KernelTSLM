# KernelSight on TimeNet

The language of the KernelSight GPU-trace dataset as it is brought into TimeNet and used to
fine-tune a time-series language model.

## Data

**Snapshot**:
One profiled run of one Variant: a 24-channel by 512-bin counter trace and its labels. One
Snapshot is one Record in TimeNet.
_Avoid_: trace, sample, recording, example

**Channel**:
One hardware counter of a Snapshot, sampled over 512 bins.
_Avoid_: sensor, feature, row

**Channel group**:
One of the five semantic groups of Channels: Pipe signature, Memory access, Discriminators,
System bandwidth, SASS modality.
_Avoid_: source group, tier

**Bin**:
One of the 512 equal-width time steps of a Snapshot. Its width is different in each Snapshot.
_Avoid_: timestep, token, frame

**Segment**:
A contiguous time interval of a Snapshot that has one workload label. Segments can overlap.
_Avoid_: phase, span, region

## Origin

**Motif**:
The family of workloads that a Snapshot comes from, such as `cutlass_gemm` or `kernelbench`.
_Avoid_: kernel type, family, source

**Problem**:
One workload definition inside a Motif, such as KernelBench `L1_P100`. All Variants of a
Problem are the same computation.
_Avoid_: kernel, task

**Variant**:
One configuration of a Problem (a dtype, a seed, or a shape) that gives one Snapshot.
_Avoid_: config, run

## Labels

**L1 workload class**:
One of the coarse workload classes, such as `matmul` or `attention`.
_Avoid_: category, bottleneck, coarse label

**L2 workload class**:
A fine workload class that refines exactly one L1 workload class.
_Avoid_: subclass, fine label

**Dominant L1 class**:
The L1 workload class that covers most of a Snapshot. It is the target of Snapshot
classification.
_Avoid_: main label, snapshot label

## Evaluation

**Grouped split**:
The partition of Snapshots into train, validation and test where all Variants of a Problem
stay in one partition.
_Avoid_: random split

**Official split**:
The train, validation and test partition that the KernelSight authors released. Variants of
one Problem can cross its partitions.
_Avoid_: original split, standard split

**Param-OOD set**:
The authors' set of Snapshots with Problem parameters outside the training range.
_Avoid_: OOD split
