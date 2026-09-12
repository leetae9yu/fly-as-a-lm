# Central-brain TinyStories comparison

This protocol tests whether the earlier optic-lobe-biased result changes when
the anatomical recurrent core is selected entirely from MaleCNS
`cb_intrinsic` neurons. It does not treat that superclass as a proven cognition
circuit, and it does not alter the original directed pairs or contact counts.

## Fixed anatomical selections

Both graphs come from the checksum-pinned full retained MaleCNS v1.0 artifact.
The source annotation and full-graph SHA256 values are recorded in
`data/central_connectome/manifest.json`.

1. Induce the graph on all 32,164 neurons whose exact superclass is
   `cb_intrinsic`.
2. Choose body 10540, the strongest incident-contact node inside that induced
   pool.
3. Expand on undirected adjacency for node selection only. Rank each newly
   reached neighbor by descending incident contacts inside the pool, then
   ascending numeric body ID.
4. Export stable-ID-sorted prefixes with every internal directed pair and its
   unchanged integer contact count. Remove boundary edges only.

No contact threshold, random sampling, training data, model seed or text metric
enters graph selection.

| Graph | Matching purpose | Neurons | Directed pairs | Contacts |
| --- | --- | ---: | ---: | ---: |
| Existing visual-biased pilot | Reference | 16,384 | 1,187,999 | 5,698,588 |
| Central edge-matched | Similar recurrent edge budget | 5,600 | 1,187,928 | 4,507,454 |
| Central node-matched | Same neuron count | 16,384 | 3,222,739 | 14,085,149 |

The edge-matched graph differs by 71 directed pairs, or 0.0060%, but has fewer
neurons. The node-matched graph has 2.71 times as many directed pairs. Running
both prevents either node count or edge count from being presented as a fully
matched regional comparison.

Prepare them locally without reading the 1.05 GB raw edge table again:

```bash
uv run --no-sync --with pyarrow==21.0.0 \
  python -m scripts.prepare_central_connectome
```

This writes:

- `data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz`
- `data/central_connectome/malecns_v1_cb_intrinsic_n16384.npz`
- `data/central_connectome/selection_order.npy`
- `data/central_connectome/manifest.json`

## Fixed training matrix

All six conditions use one T4 session, seed 0, the already frozen TinyStories
corpus, and the optimized float32 implementation. Every run starts from its own
initialization; checkpoints never cross graphs or controls.

| Graph | Control |
| --- | --- |
| Existing visual-biased N16,384 | real |
| Existing visual-biased N16,384 | shuffled |
| Central edge-matched N5,600 | real |
| Central edge-matched N5,600 | shuffled |
| Central node-matched N16,384 | real |
| Central node-matched N16,384 | shuffled |

The shuffled condition permutes target stubs deterministically while preserving
the original source and target degree multisets, edge-order initial weights,
node identities and parameter count. It may create parallel edges or self-loops.
It is a topology control, not a second anatomical graph.

Shared settings:

- corpus fingerprint
  `fd9d30ce06ef753feb2160370d40bceebd41f126aaf6be58d9d6ed42cda45575`
- train-only byte BPE vocabulary 4,096
- context 64, batch 8, 512 targets per update
- 1,000 total updates, AdamW learning rate .003, gradient clipping 1
- trainable sensory codes, 192 sensory neurons and 256 readout neurons
- held-out evaluation over every within-story target
- 64-token greedy and sampled generation from `Once upon a time`
- token-aligned full-neuron activation export for each real and shuffled run

Example single condition:

```bash
python -m flyrl.story_pilot \
  --corpus artifacts/tinystories-t4-corpus/corpus.npz \
  --graph data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz \
  --output results/central-brain/n5600-real \
  --control real --device cuda --updates 1000
```

## Comparisons and limits

The primary language outcome is held-out test NLL. Perplexity, next-token
accuracy, generation, training time and activation summaries are secondary.

- Central N16,384 versus visual-biased N16,384 is node-matched but not
  edge-matched.
- Central N5,600 versus visual-biased N16,384 is approximately edge-matched but
  not node-matched.
- Real versus shuffled isolates sensitivity to each selected graph's topology
  under the same node, edge and parameter budget.
- One seed cannot establish a regional or biological advantage.

Input and output ports remain seeded random engineering choices rather than
identified sensory or motor neurons. `cb_intrinsic` is a publisher superclass,
not a single functional circuit. Free generation quality and every failed or
divergent condition remain part of the result.

## Result status

All six conditions completed and passed recovery verification. The measured
metrics, generations, activation pages and limitations are reported in
[CENTRAL_BRAIN_RESULTS.md](CENTRAL_BRAIN_RESULTS.md).
