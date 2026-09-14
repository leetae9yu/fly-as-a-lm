# Frozen bAbI Task 1 connectome benchmark

This experiment asks one narrow question:

> Can the real-connectome recurrent model read a short synthetic story and
> generate the location answer supported by one supplied fact?

It covers English bAbI Task 1 only. It does not test Task 2, general reasoning,
general language understanding, biological function, or topology superiority.
No external RNN or attention module is allowed.

## Official source

Use Meta's current ParlAI-hosted archive:

```text
https://dl.fbaipublicfiles.com/parlai/babi/babi.tar.gz
```

The archive is pinned by both the immutable ParlAI manifest commit
`ea366da91c93f2cec8fe29b6d93b37ed96ed45bd` and exact bytes:

- size: `19,212,062` bytes;
- SHA256:
  `f7f0bee187efca0d81c3daac1b162cda4eb7f9505dee5ad6846eabbed3dbf92e`.

Only these members may enter the experiment:

| Role | Archive member | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Official train | `tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt` | 944,248 | `749ea9f7c99070feb2d88c975a254417a0dcc8274add4435ae5ae24c7afc7e9d` |
| Official test | `tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt` | 94,477 | `55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b` |
| Dataset license | `tasks_1-20_v1-2/LICENSE.txt` | 19,561 | `d12f09a636365a040fd581ebab6bf018d6fe61973c6d4862f841a6aca2f53efa` |

The embedded dataset license is CC BY 3.0. Preserve it with every redistributed
derived corpus and identify the transformation. Cite Weston et al.,
*Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*,
arXiv:1502.05698v10.

The archived bAbI generator's BSD license applies to generator software, not to
these dataset bytes.

## Parsing and split

An episode begins whenever the story-local source line ID resets to `1`.
Declarative records are `ID text`. Question records are:

```text
ID question<TAB>answer<TAB>supporting-fact-ID
```

Task 1 must have exactly one supporting fact per question. Strip incidental
outer whitespace from fields, preserve text capitalization and punctuation,
and preserve LF line separation.

The official training member contains 2,000 episodes and 10,000 questions.
Create the split before fitting or training:

1. Generate `np.random.Generator(np.random.PCG64(0)).permutation(2000)`.
2. Assign the first 1,800 episode indices to candidate train.
3. Assign the remaining 200 episodes to validation.
4. Remove a candidate-training question only when its exact formatted prompt
   appears in validation.
5. Keep source episode order and question-line order within each output split.
6. Keep the official test member's 200 episodes and 1,000 questions byte-derived
   and untouched. Test input or labels must not alter training, validation,
   initialization, hyperparameters, milestones, checkpoint selection, decoding,
   or thresholds.

This fixed rule yields:

| Split | Questions | Unique formatted prompts |
| --- | ---: | ---: |
| Train | 8,983 | 8,909 |
| Validation | 1,000 | 999 |
| Official test | 1,000 | 999 |

Seventeen candidate-training questions are excluded for exact validation-prompt
overlap. Train and validation have zero exact prompt overlap.

The untouched test happens to contain 16 questions whose exact prompt also
appears in retained train. After checkpoint selection, report the official
1,000-question score and a diagnostic score on the 984 test questions with
train-novel prompts. Do not replace the official metric or claim semantic
decontamination.

Retain official multiplicity for duplicates within each split. Repeated
questions are not independent experimental replications.

## Prompt and answer serialization

For each question, include all preceding declarative facts in its episode, in
source order. Exclude source line numbers, earlier questions and answers, future
facts, and supporting-fact IDs.

```text
Mary moved to the bathroom.
John went to the hallway.
Question: Where is Mary?
Answer:
```

The prompt ends immediately after the colon, with no trailing space or LF.
Encode the prompt and answer separately, then concatenate their IDs. The answer
segment is one leading ASCII space, the lowercase location, and one LF:

```text
 bathroom\n
```

Use the unchanged committed TinyStories byte-level BPE:

- tokenizer: `data/tinystories_quality/tokenizer.json`;
- tokenizer SHA256:
  `f626d6a0653d63580a9aad06dd5dd19bd314514dc861d271254867dfb476d4ff`;
- vocabulary: 4,096;
- implementation: `tokenizers==0.22.2`;
- no added or resized tokens.

Encode with `add_special_tokens=False`. Token 199 is the LF answer terminator.
Five locations plus LF use two target tokens; `hallway` plus LF uses three.
Reject any changed answer encoding.

This is randomly initialized Task 1 training using an externally fitted lexical
tokenizer. It is not a zero-shot test of the prior TinyStories checkpoint.

## Model and optimization

Use only:

- graph: `data/large_connectome/malecns_v1_n16384.npz`;
- graph nodes / directed edges: 16,384 / 1,187,999;
- architecture / control: `connectome` / `real`;
- seed: 0;
- 192 learned sensory-code ports and 256 disjoint linear readout ports under the
  existing `legacy_random` policy;
- trainable sensory codes, existing-edge weights, neuron biases, readout, and
  output bias;
- float32, no AMP, leak `.5`, initial gain `.9`;
- context 160, batch 8, edge chunk 65,536;
- AdamW betas `.9/.999`, epsilon `1e-8`, weight decay 0, foreach false;
- global gradient clipping 1.

Every edge endpoint remains unchanged. Load no trained model checkpoint.

Sample eight training-question indices uniformly with replacement per update
using the learner's private CPU generator seeded by the existing `seed + 37`
rule. Save sampled IDs and exact exposure counts. Evaluation and diagnostics
must not consume that generator.

Run exactly 12,000 updates, or 96,000 sampled question presentations. No
held-out-driven budget change or early stop is allowed.

For one-based update `u`:

```text
u <= 200:
    lr = .003 * u / 200
u > 200:
    lr = .0003 + .00135 * (1 + cos(pi * (u - 200) / 11800))
```

## Answer-only loss

For prompt IDs `p`, answer IDs `a`, and `s = p + a`, use `s[:-1]` as inputs and
`s[1:]` as shifted targets. Right-pad inputs to 160 with token 199.

Supervise only shifted positions `len(p)-1` through
`len(p)+len(a)-2`, inclusive. This includes the first answer token and the LF
terminator. Padding and prompt targets receive no direct loss, while gradients
still pass through the complete prompt recurrence.

Average token NLL within each question, then average the eight question losses.
This prevents the two-token spelling of `hallway` from receiving greater
per-question weight. Reject empty masks, sequence overflow, nonfinite values,
and padding-sensitive answers. Do not truncate or drop examples.

## Exact-answer evaluation and selection

Primary evaluation is unrestricted greedy generation:

1. Reset recurrent state for each example and consume only its prompt.
2. Generate from all 4,096 vocabulary tokens.
3. Stop at token 199 or after eight generated tokens.
4. Never feed a gold answer token.
5. Decode only generated IDs before the terminator.
6. Count correct only if a terminator occurred and decoded text equals exactly
   one leading space plus the gold lowercase location.

No stripping, case folding, answer-vocabulary masking, beam search, reranking,
substring extraction, or replacement output is allowed. Also report
teacher-forced per-question answer NLL and additive token-loss evidence.

Evaluate all 1,000 validation questions at QA updates:

```text
0, 500, 2000, 4000, 8000, 12000
```

Select among nonzero milestones by:

1. highest validation exact-match count;
2. lowest per-question mean answer NLL;
3. earliest update.

Freeze the selected update and checkpoint hash before opening the official test
metric. Test, train-novel-test diagnostic, and input-removal ablation may not
trigger reselection or reruns.

## Baselines and sanity checks

Fit baselines on retained train only:

1. analytic uniform answer chance, `1/6 = 16.67%`;
2. training-majority answer, lexical tie break;
3. question-person answer prior, with training-majority fallback;
4. final-fact location, ignoring the queried person;
5. a symbolic latest-location oracle.

Report validation and official-test accuracy without held-out fitting. Also
report each split's six-answer histogram and descriptive within-split majority,
without substituting a held-out majority for the train-fitted predictor.

The symbolic oracle and supporting annotation must agree on every train and
validation question before launch.

Before the main run, perform a disposable memorization check on the first two
source-ordered, two-fact training prompts per answer: 12 examples, fresh seed-0
model, 400 updates, batch 8, uniform replacement, constant LR `.003`. Require
12/12 unrestricted greedy exact answers. Discard all diagnostic state.
Failure is `SANITY_FAILED`, not permission to tune.

## Input-evidence ablation

After selecting the checkpoint, evaluate the same 1,000 test questions without
retraining. Remove every preceding fact about the queried person, including the
annotated supporting fact; keep other facts, question, answer prefix, gold
label, and decoding unchanged. Record removed source line IDs.

Let:

- `E`: ordinary official-test exact accuracy;
- `R`: support-removed exact accuracy;
- `B`: maximum of uniform chance, train-majority test accuracy, and
  question-person-prior test accuracy.

The input-evidence sanity check passes only when:

```text
E - R >= .20
R <= B + .10
```

A high ordinary score without both conditions is not credited as reading the
supplied evidence. This is an unsupported-input distribution shift and a
sanity check, not a topology audit, causal-mechanism proof, or reasoning-algorithm
identification.

## Score bands

Use exact counts out of the official 1,000-question test:

| Correct | Raw-score interpretation |
| ---: | --- |
| 950-1,000 | Conventional Task 1 solved threshold |
| 800-949 | Meaningful task learning, not solved |
| 500-799 | Weak partial task learning |
| 0-499 | Insufficient |

Report the raw-score band and input-evidence result separately.

The combined headline `Task 1 solved; input-evidence sanity check passed`
requires all 12,000 updates, all integrity and recovery gates, at least 950
ordinary test answers correct, and both ablation inequalities. If the ablation
fails, state `reading the supplied evidence was not demonstrated`.

Operational failures use `INVALID`, `SANITY_FAILED`, or
`INCOMPLETE_BUDGET_OR_INTERRUPTION`; they are not completed benchmark failures.

## T4 budget

Use one Colab Free Tesla T4 allocation. Process restart within that allocation
is allowed; allocation loss leaves the run incomplete. CPU preparation and
sealing occur first. Use one CPU thread, TF32 off, no AMP, no CPU fallback, and
record the complete runtime identity.

Before main training:

1. run exactly five disposable warmup updates and 20 timed updates at the real
   QA workload;
2. time one 1,000-example evaluation fixture including answer NLL and eight
   decoding iterations without early termination;
3. time one optimizer-bearing checkpoint write;
4. require peak allocated tensors below 10 GiB; and
5. require the conservative projection, including diagnostics, all evaluations,
   checkpointing, recovery, and 600 seconds reserve, to fit within 5,400
   allocation seconds.

Stop unfinished optimization at 6,000 allocation seconds. Recover completed
evidence and release before 6,900 seconds. This is a hard 115-minute ceiling,
not a promise that Colab will remain allocated.

## Checkpoints and recovery

Save at update zero, every 250 updates, and every validation milestone. Retain
milestone checkpoints plus the latest two complete transactions. Copy a
verified recovery bundle outside the runtime every 1,000 updates.

Each transaction binds model and Adam tensors, main sampling RNG, sampled IDs,
exposure counts, update and loss trace, next LR, immutable validation history,
and protocol/source/split/tokenizer/graph/code/runtime identities. A torn
transaction must not replace the previous valid point.

Resume restores the same state and explicitly sets the next scheduled LR.
Required evidence includes deterministic CPU-fixture uninterrupted/resumed
equality, same-T4 parameter/Adam/RNG equality, identical next sampled IDs,
reproduced greedy outputs for the first 32 source-ordered test questions in both
input conditions, and local independent recomputation of decoding, counts,
selection, score band, and ablation decision. Do not claim bitwise equality of
resumed CUDA updates to an unobserved uninterrupted counterfactual.

Publish all 1,000 ordinary and 1,000 ablated predictions with generated IDs,
termination status, gold answer, episode/question IDs, removed line IDs, and
checkpoint hash. The complete checkpoint archive may remain local if its hash
and all machine evidence needed to verify the publication are retained.

## Claim boundary

A positive result supports only:

> This one-seed, fixed-budget connectome-constrained model reached the
> conventional exact-match threshold on English bAbI Task 1, with performance
> sensitive to removal of the supplied answer evidence.

It does not establish general reasoning, general language understanding,
performance on another task, topology superiority, biological function, or a
living fly's ability to answer questions.
