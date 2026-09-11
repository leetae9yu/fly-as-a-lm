# Byte-level BPE language-model path

The autoregressive core now accepts a train-only **4,096-token byte-level BPE**
vocabulary. This is an opt-in successor to the character experiment, not a
replacement for its saved data or a new claim of language quality.

This page describes the initial implementation and small CPU check. A later
full-training-corpus comparison completed six runs before stopping; see
[BPE_RESULTS.md](BPE_RESULTS.md) and [BPE_STUDY.md](BPE_STUDY.md).

The anatomical recurrence, optimizer and learning rule are unchanged. No
attention layer or pretrained encoder was added to the anatomical model.
The later study adds a separate Transformer reference, not a replacement core.

## What changes

| Component | Character pilot | BPE path |
| --- | --- | --- |
| Prediction unit | One character | A learned subword or byte piece |
| Vocabulary | 48 | 4,096 |
| Input ports on the 16,384-neuron graph | 192 | 192 |
| Fixed code table | 48 x 192 | 4,096 x 192 |
| Readout parameters with 256 ports | 12,336 | 1,052,672 |
| Trainable parameters with the pilot graph | 1,216,719 | 2,257,055 |
| Main likelihood metrics | NLL and BPC | NLL, bits/token and token perplexity |

Smaller graphs clamp the input-port budget to half their neurons. BPE vocabulary
growth does not automatically consume more neurons as input ports. The decoder
does grow: interpreting future results must account for that increased capacity.

`alphabet_size` is retained as the historical configuration field name, but
means vocabulary size for BPE. `tokenization: "bpe"` records its interpretation.
The CLI infers both from the corpus artifact; it does not accept an unrelated
tokenizer override.

## Prepared corpus

`data/bpe_corpus/` contains:

- `corpus.npz`: versioned, pickle-free token streams with the tokenizer embedded.
- `tokenizer.json`: the identical tokenizer in Hugging Face Tokenizers format.
- `provenance.json`: source hashes, normalization, vocabulary and split details.

The tokenizer is fitted only on the first 250,000 normalized training
characters. Normalization remains lowercase plus collapsed whitespace. The
validation slice is `[0:65536]`; the fresh final-test slice is
`[131072:196608]`, after both previously evaluated character-test regions.
Offsets refer to normalized Unicode characters, before BPE encoding.

| Split | Encoded tokens |
| --- | ---: |
| Train | 58,336 |
| Validation | 17,934 |
| Test | 17,343 |

All 256 byte symbols are retained. Unseen Unicode text can be represented
without collapsing new words to one unknown ID. The literal `<unk>` markers
already present in tokenized WikiText-2 remain part of the data; this cannot
recover words that the upstream dataset had already replaced.

Preparation and serialization are deterministic, including across fresh
processes. The corpus fingerprint binds the exact tokenizer JSON, token streams,
split lengths and provenance. Changing the vocabulary, merges or corpus makes
an existing checkpoint incompatible.

```bash
python -m pip install -e '.[language]'
python -m scripts.prepare_bpe_corpus
```

The command verifies the already-present raw WikiText-2 files. It does not
download a pretrained tokenizer or a language model. The tokenizer library is
pinned to `tokenizers==0.22.2`.

Tokenizer SHA256:
`3d7bfc36989a1e522d169af0cb4273523dafbe15c02c885795ab88f27d51a07c`

Corpus fingerprint:
`df86ecf4eea9b138cf5b94d95a6644ac27c03b577a80064c072369d7634f12d7`

## Run

For a small CPU software check on the existing 16,384-neuron graph:

```bash
python -m flyrl.autoregressive \
  --graph data/large_connectome/malecns_v1_n16384.npz \
  --corpus data/bpe_corpus/corpus.npz \
  --output results/bpe-fresh --device cpu \
  --updates 32 --seeds 0 --controls real,shuffled,frozen \
  --batch-size 2 --context 8 --eval-windows 16 \
  --sample-length 32 --checkpoint-steps 16 --progress
```

Repeat the same command with `--resume --updates 34` to continue to that total
update count. Character checkpoints cannot be converted merely by replacing
their tokenizer: start a new BPE run.

For a longer T4 experiment, the existing wrapper accepts the BPE corpus:

```bash
bash run_ar_colab.sh \
  --corpus data/bpe_corpus/corpus.npz \
  --output results/bpe-t4
```

This selects the wrapper's 4,000-update budget, batch 8 and context 32, now in
**BPE tokens**. It is a starting command, not a completed BPE benchmark. It sees
more source characters per window than the old character run and revisits the
58,336-token training stream many times. Choose and record a comparison budget
before reading held-out scores.

## Read the output correctly

- `nll` is mean natural-log loss per evaluated target token.
- `bits_per_token = nll / log(2)` and `perplexity = exp(nll)`.
  If perplexity exceeds float64's range, it is null; finite NLL and bits/token
  remain available rather than crashing evaluation.
- `bits_per_character` is null for BPE. Do not compare token perplexity with
  the character pilot's BPC or published word-level WikiText perplexity.
- Sparse unigram/bigram/trigram references retain add-half smoothing and the
  same evaluation targets. They store observed contexts, not a vocabulary cube.
  These are simple references; large-vocabulary add-half smoothing is not a
  tuned subword language baseline.
- `generated_token_ids` preserves exact prompt, sampled and greedy token IDs.
- `generated_with_prompt` decodes prompt and continuation IDs together. Use it
  for complete text: a prompt can end between the bytes of a Unicode character.
- `generated` retains separate prompt/continuation previews for compatibility.
  Arbitrary generated invalid UTF-8 can appear as replacement characters.
  BPE vocabulary labels themselves are never concatenated as display text.

Scores from the short CPU check establish an executable path, not useful BPE
language quality. The earlier 45.21% accuracy result remains a **character**
result. A longer, matched BPE experiment and GPU-specific BPE validation are
separate from this implementation check.

## Verification and compatibility

The CPU implementation check used the actual 16,384-neuron graph, 4,096 BPE
tokens, seed 0, batch 2, context 8, and all three controls. Each ran 32 updates,
then resumed to 34, with 16 evaluation windows and 32 generated tokens.
This is only 544 training targets per condition, not a language-quality study.
Records are under `results/bpe-smoke/`.

Separate verification copies continued from 34 to 38. Every condition reproduced
the stored generation and restored parameters, Adam state, RNG and trace exactly.
Continuation differences were zero on CPU. The source checkpoints were unchanged.
Evidence is under `results/bpe-resume-check/`. A pre-BPE character checkpoint
also retained exact CPU continuation under the updated code.

Tests cover train-only fitting, merge-tie determinism across processes, unseen
Unicode and whitespace, artifact tampering, fixed input-port counts at larger
vocabularies, sparse/dense n-gram equivalence, bounded baseline allocation,
decoded generation, UTF-8 prompt boundaries and exact CPU continuation.

The existing character corpus format remains strict and unchanged. Its
configuration limits remain in force, and older character checkpoint metadata
loads with the original character interpretation. Recorded character artifacts
are not rewritten.

```bash
python -m scripts.verify_ar_resume \
  --graph data/large_connectome/malecns_v1_n16384.npz \
  --corpus data/bpe_corpus/corpus.npz \
  --run results/bpe-fresh/seed-0/real \
  --output results/bpe-resume-check
```

Data retains the WikiText licensing and attribution described in
[THIRD_PARTY.md](THIRD_PARTY.md) and [data/wikitext2/README.md](data/wikitext2/README.md).
