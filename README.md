# FLY AS A LANGUAGE MODEL

A character language model built on a real fruit-fly connectome. Text stimulates
fixed sensory codes; activity propagates through anatomical connections; a small
readout predicts the next character. Backpropagation changes the weights on those
connections, without replacing the recurrent core with a transformer.

**Status: held-out character prediction learned; fluent language and a
biological-wiring advantage not demonstrated.** The first T4 pilot reached
45.21% next-character accuracy. A rewired control reached the same accuracy.
The full retained graph fits on T4; natural-text training used a 16,384-neuron
subset. Results, controls, generated text and assumptions are included.

Inspired by [DOOMFLY](https://github.com/nftechie/doomfly), this independent
experiment takes the connectome-to-computation question from game control to
text prediction.

**Subword option:** a train-only 4,096-token byte-level BPE path is available.
See [BPE.md](BPE.md) for preparation, training and token-level metrics. The
loop and measured scores below describe the original **character** pilot;
they are not BPE performance results.

## The loop

1. A character from a **48-character alphabet** activates a fixed bipolar code
   on **192 sensory neurons**. These ports are sampled, not identified natural
   language or sensory circuits.
2. Continuous neural states propagate through **16,384 neurons and 1,187,999
   directed connections** retained from MaleCNS v1.0. Only existing anatomical
   edges carry learned recurrent weights.
3. A **256-neuron readout**, disjoint from the input ports, predicts the next
   character. Its 12,336 parameters form a small decoder, not a pretrained
   language model.
4. Teacher-forced next-character loss trains edge weights, neuron biases and
   the readout through **32-character windows**. The topology stays fixed.
5. During free generation, each predicted character becomes the next input.
   Neural state persists; no future reference text is supplied.

```text
character -> fixed sensory code -> anatomical recurrent network -> next character
                                     ^                               |
                                     +-------- feedback -------------+
```

The wiring comes from a biological reconstruction. The scalar neuron dynamics,
functional signs, input/output ports and learning rule are engineering choices.
This does not demonstrate that a living fly can learn human language.

## First experiment

One seed, three matched conditions, 4,000 updates each on Colab Free with a
Tesla T4. Each condition sees 1,024,000 next-character training targets from a
250,000-character WikiText-2 training slice.

All models and train-fitted n-gram baselines are evaluated on the same **1,024
held-out target positions**. Lower bits per character (BPC) is better.

| Model | Test accuracy | Test BPC |
| --- | ---: | ---: |
| Unigram | 18.36% | 4.3614 |
| Bigram | 31.45% | 3.3275 |
| Trigram | 41.02% | **2.8048** |
| Anatomical, trained | **45.21%** | 2.8057 |
| Rewired, trained | **45.21%** | 2.8253 |
| Anatomical, frozen core | 28.81% | 3.6720 |

Training the anatomical model reduced its test BPC from **5.5852 to 2.8057**.
It improved accuracy over the trigram, but not BPC. Rewiring produced nearly
the same result. One initialization is not evidence of anatomical superiority.

The rewired control preserves directed degree counts, permits parallel edges
and self-loops, and retains the original edge-order initial weights. The frozen
control trains only the readout. [Full results](AR_RESULTS.md) include the
initial scores, core ablation, timings and checkpoint verification.

### What it writes

Anatomical model, sampled continuation. The actual training-only prompt is
`= valkyria chronicles iii = senj`. Output is wrapped for display:

```text
el arment some e kordungsesy the have to reage poluches nov reperateen
thas sere hed at roptrale hith relayp , semunn anced whe eimol smankay
an fhelait cormecration , surcy nergo . " h werve under hir hule wor
sernom nemy ; b1 a follen gs
```

It has learned character patterns and fragments of words, not coherent prose.
Greedy generation enters repetitive loops. The unedited outputs are in the
[anatomical model report](results/ar-main/seed-0/real/report.json).

## The scale boundary

MaleCNS v1.0 covers the brain **and ventral nerve cord**. The full retained
annotated neuronal graph has **166,700 neurons, 25,582,938 directed pairs and
124,177,617 synaptic contacts**.

| Graph | What was tested | T4 warm update | Peak allocated GPU memory |
| --- | --- | ---: | ---: |
| 16,384 neurons | Natural-text training and matched controls | 0.188 s | 0.197 GiB |
| 166,700 neurons | Synthetic forward/backward/AdamW capacity | 4.626 s | 2.591 GiB |

These short capacity measurements use batch 8 and context 32; GPU allocation
excludes driver overhead. The full graph was **not trained on natural text**.
The trained subset is an optic-lobe-biased induced hub-BFS neighborhood, not a
whole brain or a uniformly sampled miniature brain.

Cached sparse patterns avoid repeated sorting and any dense neuron-by-neuron
matrix. This brought full-graph update time down from 11.872 to 4.626 seconds.
See [anatomy and provenance](data/large_connectome/README.md) and the
[recorded capacity measurements](artifacts/ar-capacity-cached.json).

## Run the experiment

Use a Colab T4 runtime with a compatible preinstalled PyTorch, or a local CUDA
machine. The pilot graph and corpus are committed; the 1.07 GB raw connectome
download is not needed for this run.

```bash
git clone https://github.com/leetae9yu/fly-as-a-lm.git
cd fly-as-a-lm
python -m pip install -e '.[language]'
bash run_ar_colab.sh --output results/ar-fresh
```

The recorded run used Python 3.13, Torch 2.11.0+cu128 and NumPy 2.5.3. Local
checks used Python 3.12. The three measured run calls took about 30 minutes
combined, excluding environment setup. Your hardware and software may differ.

For a small **CPU software smoke test**, not a reproduction of the pilot:

```bash
python -m flyrl.autoregressive \
  --graph data/large_connectome/malecns_v1_n256.npz \
  --corpus data/ar_corpus/corpus.npz \
  --output results/cpu-smoke --device cpu \
  --updates 10 --seeds 0 --controls real \
  --batch-size 2 --context 8 --eval-windows 16 --sample-length 80
```

Continue a run to a **total** update count:

```bash
bash run_ar_colab.sh --output results/ar-fresh --resume --updates 5000
```

Checkpoints are written every 200 updates. Parameters, Adam state, window RNG
and progress are restored; incompatible configurations and runtimes are
rejected. CUDA continuation can have small numerical differences. Colab VM
storage is temporary: download checkpoints before releasing a runtime.

Model weights and large raw downloads are not stored in Git. Re-running the
commands creates your own checkpoints. The committed JSON reports preserve
the original measurements. Detailed setup and verification:
[AUTOREGRESSIVE.md](AUTOREGRESSIVE.md).

## Work on this with me

The useful next step is to test which parts of the result survive stronger
controls, not to turn the current score into a claim about fly intelligence.

- **Replication:** run more seeds under the same budget and report the spread.
- **Baselines:** compare against parameter-matched conventional and sparse RNNs.
- **Anatomy:** test other regions and input/output placements with matched controls.
- **Scaling:** improve sparse throughput and measure longer or larger runs.
- **Interpretability:** inspect what changes in the trained circuit, using
  interventions rather than labels inferred from attractive visualizations.

For experimental contributions, include configuration, seed, graph/corpus
hashes, train/validation/test separation, matched controls and machine-readable
results. Keep negative results. For implementation changes, include a regression
test or a reproducible numerical check.

```bash
python -m pip install pytest
python -m pytest -q
```

The suite has 142 passing tests, including sparse derivatives, BPE fitting and
decoding, causality, controls and CPU checkpoint continuation. The character
pilot also had CUDA derivative and checkpoint checks; software correctness is
not biological validity.

## Repository map

| Path | Contents |
| --- | --- |
| `flyrl/ar_*.py`, `flyrl/autoregressive.py` | Sparse character LM, training, evaluation and checkpoints |
| `flyrl/bpe_*.py`, `data/bpe_corpus/`, [BPE.md](BPE.md) | Train-only byte BPE, prepared inputs and subword usage |
| `scripts/prepare_large_connectome.py` | Pinned anatomical data importer |
| `data/large_connectome/`, `data/ar_corpus/` | Prepared inputs, source records and hashes |
| `results/ar-main/`, `results/ar-resume/` | Measured results and restoration evidence |
| `tests/` | Numerical, learning, data and checkpoint checks |
| [AR_RESULTS.md](AR_RESULTS.md) | Full pilot results and limitations |
| [AUTOREGRESSIVE.md](AUTOREGRESSIVE.md) | Model definition and reproduction protocol |
| [PROTOTYPE.md](PROTOTYPE.md), [LANGUAGE.md](LANGUAGE.md) | Earlier reward-learning experiments |

## License and sources

Original project code is [MIT licensed](LICENSE), copyright **leetae9yu**.
Connectome data, WikiText and copied upstream reference files retain their own
licenses and attribution; see [THIRD_PARTY.md](THIRD_PARTY.md). This is an
independent experiment, not an official MaleCNS or DOOMFLY project.
