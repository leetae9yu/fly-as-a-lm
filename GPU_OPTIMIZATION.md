# Anatomical GPU optimization

The first optimization pass was **3.11 times faster** than its unchanged
same-T4 baseline. A second pass then made that selected implementation another
**1.46 times faster** in a fresh same-T4 comparison. It preserves the graph,
parameter shapes, float32 precision and recurrence equations. This is a
computation optimization, not a new language-quality result or an enlarged-brain
experiment.

## Same-hardware comparison

All variants used one Tesla T4 with Torch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0
and NumPy 2.1.3. The workload was the existing TinyStories pilot: 16,384 neurons,
1,187,999 original edges, vocabulary 4,096, context 64, batch 8 and learnable
sensory codes.

Each repetition restored the same published 1,000-update checkpoint and private
sampling RNG, warmed up for three updates, then timed ten updates with CUDA
synchronization at the boundaries. The table reports the median of three
repetitions. Corpus sampling and the optimizer are included; checkpoint I/O,
initialization, JIT compilation, profiling and evaluation are excluded.

| Variant | Seconds/update | Speedup versus baseline |
| --- | ---: | ---: |
| Unchanged baseline | 0.397400 | 1.00x |
| Batched sequence readout only | 0.402911 | 0.99x |
| Fused edge gradients only | 0.136207 | 2.92x |
| **Both, selected** | **0.127602** | **3.11x** |

Per-repetition measurements:

| Variant | Repetition 1 | Repetition 2 | Repetition 3 |
| --- | ---: | ---: | ---: |
| Baseline | 0.397400 | 0.393718 | 0.398699 |
| Readout only | 0.400375 | 0.402911 | 0.407675 |
| Edge gradients only | 0.134803 | 0.136207 | 0.147580 |
| Both | 0.127602 | 0.126921 | 0.127934 |

The selected version processes about **4,012 training targets/second**, versus
1,288 before. Peak allocated CUDA tensors decreased from 270.13 to 262.14 MiB.
These are allocated tensors, not total process memory or GPU reservation.
The small readout-only timing difference is not a demonstrated standalone
speed improvement; gradient fusion supplies the main gain. The combined
version was faster than kernel-only in this short comparison, not a claim
about every workload or device.

## Second pass: sparse recurrence

The second pass used a newly allocated T4 and a longer timing protocol. Each
variant restored the same published checkpoint and RNG, warmed for five updates,
then measured 20 updates. The table reports the median of five repetitions.
JIT compilation, loading, evaluation, checkpoint I/O and profiling were outside
the timed interval.

| Variant | Seconds/update | Targets/second | Speedup |
| --- | ---: | ---: | ---: |
| First-pass implementation, remeasured | 0.115854 | 4,419 | 1.00x |
| Sequence-prepared native COO | 0.091515 | 5,595 | 1.27x |
| **Sequence-prepared Triton CSR, selected** | **0.079132** | **6,470** | **1.46x** |

| Variant | R1 | R2 | R3 | R4 | R5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current baseline | .121205 | .113612 | .114204 | .115854 | .120405 |
| Prepared COO | .091515 | .090853 | .094288 | .097201 | .091302 |
| Triton CSR | .097785 | .079132 | .079010 | .073360 | .089622 |

The Triton measurements vary more than the native ones, so every repetition is
shown. The 1.46x result is the controlled comparison for this pass. The first
and second passes used different T4 sessions and timing protocols: multiplying
3.11x by 1.46x gives an indicative **4.56x cumulative improvement**, not one
paired benchmark. Dividing the first pass's original .3974 seconds by the final
.07913 seconds gives 5.02x, but that cross-session ratio is less controlled and
is not the headline result.

## What changed

### Sequence readout

`ConnectomeLM._advance` performs the unchanged neural state update. Online
`step` still projects each state immediately. Sequence `forward` collects the
causal readout states and projects them in one matrix multiplication rather
than launching a separate dense multiplication per token.

No attention, gates, connections or contextual encoder were added. Recurrent
time dependence remains sequential. This does not promise faster online
generation, which continues to use `step`.

### CUDA edge-weight gradients

Previously, every timestep processed 19 chunks of original edges. Each chunk
materialized two gathered edge-by-batch arrays, multiplied and reduced them,
then copied the result into the edge-gradient vector.

`flyrl/cuda_edge_grad.py` fuses this operation for CUDA float32. Each Triton
program owns 128 original edges and reduces batch tiles of at most 32 columns
in registers. It stores one gradient per original edge without materializing
full edge-by-batch intermediates. Unsorted edges, duplicates, self-loops,
strided tensors and the current CUDA stream are supported.

In the first pass, forward propagation and state gradients still used native
sparse multiplication. CPU and other previously supported dtypes retain the
bounded native edge-gradient path. Import, compilation or launch errors are
not silently converted into a CPU or alternative-backend run.

### Sequence-scoped sparse preparation

The second pass orders and coalesces current edge weights once per input
sequence, then constructs forward and reverse sparse operators from that
immutable snapshot. Previously this work occurred at every token in forward
and backward.

The custom backward still emits one derivative per **original** edge, so
parallel parameters created by shuffled controls remain independent. Named
parameters, Adam slots, checkpoint arrays and graph fingerprints are unchanged.
Prepared tensors live for one call; CSR row pointers and columns are derived,
nonpersistent topology buffers.

In the profiled update, preparation reduced `aten::index` from 384 calls and
22.40 ms of device time to 194 calls and .90 ms. Native sparse multiplication
and its 128 COO-to-CSR conversions remained at this intermediate stage.

### Triton CSR sparse multiplication

`flyrl/cuda_sparse_mm.py` consumes cached CSR layouts and one sequence's
prepared values for CUDA float32 recurrence. One Triton program owns an output
row and up to 32 batch columns; edge products stay in bounded register tiles.
Forward and transpose use the same primitive with their corresponding layouts.

Native cuSPARSE's 128 matrix products accumulated 37.84 ms after sequence
preparation. The selected Triton `multiply` kernel accumulated 18.57 ms across
the same 128 calls. Total recorded CUDA device events fell from 4,033 in the
current baseline to 2,691. CPU and non-float32 recurrence retain prepared native
COO. Backend failures are not silently hidden by a fallback.

## Measured bottleneck

This comparison includes actual CUDA traces, not inferred GPU percentages from
the earlier CPU profile. In one profiled baseline update, the vectorized gather
kernel ran 2,560 times and accumulated about 250.4 ms of device time. After
optimization, the fused edge-gradient kernel ran 64 times and accumulated
about 6.7 ms.

The complete Chrome traces distinguish execution from transfer events:

| Trace category | Baseline | Selected |
| --- | ---: | ---: |
| Kernel events | 9,266 | 4,025 |
| GPU memcpy events | 1,220 | 4 |
| GPU memset events | 0 | 3 |

The raw summaries' historical `kernel_calls` field counts **all CUDA device
events**, including transfers/annotations: 10,487 before and 4,033 after. It
must not be interpreted as a pure kernel count. The table above was derived
from the trace categories themselves.

At the end of the first pass, the remaining major device operations included
native sparse matrix products, weight/index rearrangement and copies. The
second pass targets the first two. Pointwise state updates, input/readout
indexing and sequential dependence between tokens remain.

## Correctness and compatibility

- The readout performance test first observed four dense projections where one
  was required; it passes after batching. Independent dense equations verify
  logits and gradients for real, shuffled and frozen circuits, including the
  zero-recurrence diagnostic.
- The CUDA candidate tests first failed because the implementation was absent.
  They cover batches 1, 3, 8, 17, 33, 65 and 129; irregular strides; duplicate
  edges; self-loops; tile tails; zero dimensions; broadcast strides; and a
  non-default CUDA stream.
- An integration test first observed four materialized edge gathers in native
  backward, then zero through the optimized sparse recurrence.
- Same-input fixtures compare every logit and every trainable parameter
  gradient for both an initialized model and the published trained checkpoint.
  In the selected implementation, maximum trained-logit difference was
  **5.72e-6** and maximum trained-parameter-gradient difference **4.06e-8**.
  The tolerances were fixed before comparison: logit rtol 1e-4, gradient rtol
  1e-3, and atol 2e-5.
- The same 16-token greedy continuations were reproduced for both fixtures.
- Final focused T4 suite: **34 passed**. Local full suite: **225 passed,
  19 CUDA-only skips**. Ruff, changed-file strict typing and package build passed.
  The previously reported whole-project NumPy typing issues in untouched
  `tests/test_large_connectome.py:26-27` were outside this change.
- The public GPU CLI resumed a disposable checkpoint copy from update 1,000 to
  1,002 and exported activations. The original checkpoint hash stayed unchanged.
- The second-pass isolated and integrated T4 suites passed **76 tests**. They
  cover irregular CSR degrees, empty rows, batches 1/3/8/17/32/64, strided and
  broadcast inputs, non-default streams, validation failures and bounded
  workspace. The integration test first observed zero CSR dispatches and then
  exactly one forward and one transpose dispatch.
- Multistep dense references cover original-edge gradients with duplicate
  edges, separate prepared snapshots and trailing empty nodes. CSR buffers do
  not enter `state_dict`.
- Second-pass initialized/trained fixtures retained the same losses, 16-token
  greedy generations and all-parameter gradients within the original
  tolerance. Maximum aggregate trained difference was **4.77e-6**.
- A 100-update comparison consumed identical training windows from the same
  update-1,000 checkpoint. Maximum per-update training-NLL difference was
  .00465; validation/test NLL differences were .00235/.000325. Parameters
  diverged by as much as .261 and 32-token greedy generations differed. This
  supports matched short-run metrics, **not** trajectory or generation identity.
- Final local suite after production integration: **232 passed, 54 CUDA-only
  skips**; strict lint/types, no-excuse audit and package build passed. Final
  refactored T4 integration suite: **76 passed**.

Floating-point reduction order can differ. Bitwise-identical training
trajectories are not promised, and no new long training run was used to claim
unchanged final language quality.

## Environment and evidence

For a fresh compatible CUDA environment:

```bash
uv sync --extra language --extra cuda
```

The optional `cuda` extra declares Triton. CPU import and training do not require
it. On Colab, preserve its working CUDA PyTorch installation; the tested runtime
already included the matching Triton 3.6.0. Other GPU architectures were not
benchmarked.

Local evidence:

- `artifacts/gpu-opt-{baseline,readout,kernel,combined}.zip` and extracted
  directories: measured summaries, compressed Chrome traces and baseline
  numerical fixtures.
- `artifacts/gpu-opt-cli.json`: actual checkpoint-resume entry-point verification.
- `artifacts/gpu-opt2-{current,prepared-coo,triton-spmm}.zip` and extracted
  directories: second-pass numerical checks, five timing rounds and traces.
- `artifacts/gpu-opt2-trajectory.zip`: both 100-update summaries and comparison.
- `artifacts/tinystories-t4-input.zip`: immutable pre-optimization source/inputs.
- `.omo/evidence/gpu_opt_benchmark.py`: the fixed benchmarking procedure;
  `.omo/evidence/gpu_opt_execute.py` runs one named variant on the remote VM.

The published source checkpoint SHA256 is unchanged:
`e0ad8077ceabf61875e4cf38faffc95e781c1535f6688ed544c1269441a4b99a`.
Graph size, training corpus and tokenizer match [the completed pilot](TINYSTORIES_RESULTS.md).
The optimization T4 was released after evidence recovery.
