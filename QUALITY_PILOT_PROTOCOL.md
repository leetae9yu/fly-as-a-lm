# TinyStories Quality-1 protocol

This prospective experiment prioritizes language quality and demonstration
quality. It does not test whether biological wiring is superior to a randomized
topology.

## Model boundary

- Graph: `data/large_connectome/malecns_v1_n16384.npz`
- Core: 16,384 scalar recurrent states and 1,187,999 original directed edges
- Input: 192 learned sensory-code ports
- Output: 256 disjoint linear-readout ports
- No Transformer, GRU, attention, pretrained model, distillation or external
  contextual encoder
- Fresh initialization with seed 0

The existing sparse anatomical recurrence and every original edge endpoint remain
unchanged.

## Corpus

- Source: original `roneneldan/TinyStories`
- Revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Fetch bounded prefixes of 51,000 complete official-training stories and 800
  complete official-validation stories
- Select the first 50,000 unique normalized training stories
- Exclude every selected training identity from the official-validation pool
- Select the first 256 remaining unique stories as validation and the next 512
  as test
- Strip outer whitespace only and exact-deduplicate normalized UTF-8 text
- Fit a fresh 4,096-token byte-level BPE vocabulary only on selected training
  stories with `tokenizers==0.22.2`
- Encode stories separately and prohibit cross-story training or evaluation pairs

The prefixes are deliberately bounded and are not representative random samples
of the full source dataset. Exact deduplication does not exclude paraphrases.
Source hashes, consumed byte counts, story identities, offsets, tokenizer and
license text are frozen before GPU allocation.

## Training

- One real-connectome condition
- Float32 on one Tesla T4; CPU fallback is forbidden
- Context 128, batch 8
- 30,000 updates, totaling 30,720,000 sampled next-token targets
- AdamW betas `.9/.999`, epsilon `1e-8`, weight decay 0
- Gradient clipping 1, leak `.5`, initial gain `.9`
- Train sensory codes, recurrent edge weights, neuron biases and linear readout

For one-based update `u`, the learning rate is:

```text
u <= 500:
    0.003 * u / 500
u > 500:
    0.0003 + 0.00135 * (1 + cos(pi * (u - 500) / 29500))
```

There is no heldout-driven budget extension or early termination. Checkpoints
are written every 1,000 updates.

## Evaluation and selection

Evaluate the complete validation split at updates:

```text
0, 1000, 5000, 10000, 15000, 20000, 25000, 30000
```

State persists within each story and resets between stories. Every within-story
next-token pair is scored exactly once; first tokens and cross-story transitions
are excluded.

After update 30,000, choose the nonzero evaluated checkpoint with the lowest
validation NLL. Exact ties choose the earlier checkpoint. Evaluate update 1,000,
the selected checkpoint and update 30,000 on the untouched test split,
deduplicating identical checkpoints. Report token-weighted NLL, BPE perplexity,
accuracy, denominators and per-story additive evidence.

Train-fitted unigram and within-story bigram references are descriptive only.
New-tokenizer perplexities are not directly compared with historical runs.

## Generation panel

Use these exact prompts:

1. `Once upon a time`
2. `Lily found a small red box under her bed.`
3. `Ben wanted to fly his kite, but there was no wind.`
4. `A little rabbit lost the key to her house.`
5. `"Can I play with you?" asked Tom.`
6. `Mia promised to look after her brother's toy.`
7. `The rain stopped, and the children opened the door.`
8. `A small bird was afraid to leave its nest.`

Generate 192 new tokens while retaining recurrent state and feeding back only
generated tokens.

- At update 1,000 and update 30,000: one greedy and one sampled continuation per
  prompt
- At the selected checkpoint: one greedy and three sampled continuations per
  prompt
- Sampling temperature `.8`, nucleus probability `.9`, no top-k, repetition
  penalty, blocking, reranking or replacement samples
- Sample seed for zero-based prompt `i` and zero-based draw `j`:
  `10000 + 100*i + j`

Publish every output and token ID sequence, including failures.

## Activation deliverables

For prompts 1 and 4 at update 1,000 and the selected checkpoint:

- full `192 x 16,384` float32 state arrays;
- original neuron IDs, token IDs, selected-token probabilities, causal contexts,
  ports and artifact fingerprints;
- magnitude-ranked and temporal-variation-ranked 64-neuron heatmaps on a fixed
  `[-1, 1]` scale; and
- whole-recording saturation and temporal-variation statistics.

For the selected checkpoint and prompt 1, render a fixed-camera anatomical
playback of the first 64 tokens at 3 fps. These are computational model states,
not biological firing, attention or causal importance.

## Launch and recovery gates

Before training:

- verify source, split, tokenizer, graph and legal-window identities;
- run actual-T4 sparse numerical tests and a finite update;
- benchmark five warmups plus twenty updates;
- require peak allocated tensors below 10 GiB and projected total runtime below
  3.5 GPU-hours.

At most two T4 allocations and four aggregate GPU-hours may be used solely to
finish this fixed run. Resume requires identical protocol, source, graph,
corpus, checkpoint and runtime identities. Recover and independently verify
checkpoints, Adam moments, window RNG, progress, learning-rate position, metrics,
generation, activation arrays and figures before releasing the final runtime.

## Reporting decision

Report `language-quality improvement demonstrated` only if:

1. all 30,000 updates and recovery gates complete;
2. selected-checkpoint test NLL improves over update 1,000 by at least `.50`
   nat/token; and
3. at most 3 of the 24 selected sampled continuations contain a 1-10-word block
   repeated consecutively at least three times, using lowercase words matched by
   `[a-z]+(?:'[a-z]+)?`.

Otherwise report the completed metrics and generations without the improvement
label. In particular, if likelihood improves but generation remains unreliable,
state:

> Held-out prediction improved; reliable short-story generation was not
> demonstrated.
