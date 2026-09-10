# Verified MaleCNS v1.0 anatomy

These are real public adult male fly connectome artifacts, not synthetic fixtures.
The release covers brain **and ventral nerve cord**. `full` means the full retained
annotated neuronal graph under the policy below, not every segmentation object,
not a complete living fly, and not a language model by itself.

## Artifacts

All paths are relative to `data/large_connectome/`. Exact SHA256s, download links,
contact counts, class composition, and retention fractions are in `manifest.json`.

| NPZ | Neurons | Directed pairs | Bytes | Full-retained neurons / edges |
| --- | ---: | ---: | ---: | ---: |
| `malecns_v1_n256.npz` | 256 | 6,153 | 22,194 | 0.1536% / 0.0241% |
| `malecns_v1_n1024.npz` | 1,024 | 17,596 | 62,837 | 0.6143% / 0.0688% |
| `malecns_v1_n4096.npz` | 4,096 | 82,836 | 303,610 | 2.4571% / 0.3238% |
| `malecns_v1_n16384.npz` | 16,384 | 1,187,999 | 4,374,941 | 9.8284% / 4.6437% |
| `malecns_v1_full.npz` | 166,700 | 25,582,938 | 105,761,677 | 100% / 100% |

Full graph extraction was feasible and completed on the four-core ARM host.
The raw files total 1,065,725,260 bytes. Batch parsing avoids expanding the
151,856,684 source rows into Python edge objects. Full retained anatomy has
124,177,617 synaptic contacts, 217 isolates, and 101 self-edges.

The importer reproduced the upstream counts independently: 211,577 annotation
rows, excluding 11,864 explicit glia and 33,013 unresolved objects; 126,273,746
source edge rows excluded because at least one endpoint was not retained.
The retained graph contains 78.7893% of annotation rows and 16.8468% of released
edge rows. Those denominators include non-neuronal/unresolved segmentation
objects and must not be interpreted as the fraction of a living brain modeled.

## Source, license, and conventions

- Source: https://male-cns.janelia.org/download/
- Paper: https://doi.org/10.1016/j.cell.2026.08.015
- Attribution: MaleCNS Consortium / HHMI Janelia Research Campus, MaleCNS v1.0.
- Data license: **CC-BY-4.0**; see `LICENSE.CC-BY-4.0.txt`.
- Discovery reference: https://github.com/nftechie/doomfly at commit
  `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`. Its dataset registry, original source
  lock, and normalized report are preserved in `upstream/`; no upstream code was
  executed. `upstream/download.html` records the publisher's license statement.
- `raw/annotations.feather` SHA256:
  `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`.
- `raw/edges.feather` SHA256:
  `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`.

Retain every nonempty assigned neuronal superclass, including uncertain `tbc`
classes, excluding explicit `status == Glia`; do not restrict tracing status,
cell type, or region. Preserve all released edges between these nodes, including
single-contact edges and autapses. The release already uses pre/post confidence
threshold 0.5. Duplicate directed pairs are aggregated by exact integer addition;
the verified release had zero duplicate retained pairs.

`body_pre` is presynaptic source and `body_post` postsynaptic target. Stable IDs
are `malecns:v1.0:<exact decimal bodyId>`, sorted numerically without float
conversion. NPZs use the existing Graph keys and add `synapse_count` (int64).
`synapse_count` is anatomical contact count. `weight` is float64 `log1p(count)`,
an all-positive **model initialization**, not measured efficacy or transmitter
sign. The neurotransmitter file is neither required nor downloaded; its original
pin remains in the upstream lock for source identity only.

## Subgraph selection and bias

The deterministic root is body **10009**, the retained neuron's maximum total
incoming plus outgoing contact count. Root ties use ascending bodyId. Breadth-
first expansion uses undirected adjacency only for node selection; each node's
new neighbors are ordered by descending global incident contacts, then ascending
bodyId. `selection_order.npy` stores the exact bodyId discovery order. Each size
is a prefix of that order, exported with stable-ID sorting and **all** internal
directed edges and unchanged contact counts. Boundary edges are removed.

Every prefix is weakly connected, not necessarily strongly connected. This is a
hub-neighborhood selection, not uniform sampling or a named anatomical region.
It is strongly optic-lobe biased: 15,355 of the 16,384-node subset are
`ol_intrinsic`, versus 89,403 of 166,700 in the full retained graph. The small
subsets retain proportionally few full-graph edges, so node count alone is not a
fair proxy for computational cost. Use measured edge counts for scaling claims.
The 256-node artifact is a manageable control graph; do not invoke the existing
Python-object `shuffled_graph` routine on the 25.6M-edge graph.

## Reproduction

From the project root, without modifying the shared environment:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync --with pyarrow==21.0.0 \
  python -m scripts.prepare_large_connectome
uv run --no-sync pytest tests/test_large_connectome.py tests/test_connectome.py -q
```

Missing raw files download automatically over HTTPS. Existing raw files must
match built-in byte counts and SHA256 pins; corruption fails rather than being
accepted or silently replaced. `--no-download` enforces an offline cached import.
`--output PATH` selects another artifact directory. Only the importer needs
PyArrow; loading the shipped NPZs and the synthetic unit tests do not.
Tests use explicitly synthetic fixtures solely to test direction, exact IDs,
duplicate aggregation, induced extraction, connectivity, and source integrity.
No scientific model training is performed by this importer.
