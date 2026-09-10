# Third-party sources and license scope

The root MIT license covers original project code. It does not relicense data,
upstream documentation or other third-party material.

| Material | Source and attribution | Terms and local record |
| --- | --- | --- |
| MaleCNS v1.0 anatomy and derived graph arrays | MaleCNS Consortium / HHMI Janelia Research Campus; [publisher](https://male-cns.janelia.org/download/), [paper](https://doi.org/10.1016/j.cell.2026.08.015) | CC BY 4.0; [license](data/large_connectome/LICENSE.CC-BY-4.0.txt), [transformations and hashes](data/large_connectome/README.md) |
| WikiText-2 raw text and derived character/BPE corpora | Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher; Wikipedia contributors; [dataset card](https://huggingface.co/datasets/Salesforce/wikitext) | Upstream Attribution-ShareAlike/GFDL notices remain applicable; [scope and preprocessing](data/wikitext2/README.md), [source hashes](data/wikitext2/provenance.json), [character test slice](data/ar_corpus/provenance.json), [BPE provenance](data/bpe_corpus/provenance.json) |
| Larval mushroom-body data used by the historical prototype | Eichler et al., [Nature 2017](https://doi.org/10.1038/nature23455); distributed by graspologic | [Upstream distribution license](data/LICENSE.graspologic.txt), [provenance](data/provenance.json) |
| Copied MaleCNS registry, source lock and normalized report in `data/large_connectome/upstream/` | [nftechie/doomfly](https://github.com/nftechie/doomfly/tree/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33), at the pinned commit | [Upstream MIT notice](data/large_connectome/upstream/LICENSE.doomfly.txt); these reference files helped locate and verify the dataset |

The WikiText card lists CC BY-SA 3.0/GFDL in metadata and links CC BY-SA 4.0 in
its licensing discussion. This project preserves that upstream distinction
rather than declaring the corpus MIT. Follow the linked upstream terms when
redistributing raw or transformed text.

`data/large_connectome/upstream/download.html` is a publisher-page snapshot
retained as source/license evidence, not original project code.

DOOMFLY inspired the experiment and the README's concise experiment-first
presentation. This repository is an independent implementation and makes no
claim of affiliation or endorsement. Upstream attribution is separate from
authorship of this project's Git commits.
