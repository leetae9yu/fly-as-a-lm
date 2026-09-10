# WikiText-2 character task

This directory contains the tokenized WikiText-2 files distributed in
[pytorch/examples](https://github.com/pytorch/examples/tree/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model/data/wikitext-2).
The official train/validation/test assignments are retained.

WikiText was assembled from Wikipedia Good and Featured articles by Stephen
Merity, Caiming Xiong, James Bradbury, and Richard Socher. See
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) and the
[upstream dataset card](https://huggingface.co/datasets/Salesforce/wikitext).
The text is distributed under Creative Commons Attribution-ShareAlike terms,
not the code's license. The upstream card's metadata lists CC BY-SA 3.0/GFDL
and its licensing section links CC BY-SA 4.0; retain the upstream attribution
and licensing information when redistributing the text.

`raw/` retains the unmodified download. `provenance.json` records the exact
commit, SHA-256 checksums, transformations, alphabet, and split lengths.
`scripts/prepare_language.py` verifies those checksums before preprocessing.

Changes made for this experiment:

- Lowercase and collapse whitespace independently within each split.
- Keep the first 250,000 normalized training characters and the first 65,536
  normalized characters from each held-out split.
- Fit a 48-character vocabulary on the retained training text only.
- Map other characters to `?` (index zero). Genuine question marks share that
  index, so `unknown_token_fractions` slightly overstates out-of-vocabulary rate.
- Retain the source's `<unk>` and `@-@` tokenization artifacts. This is not the
  separate WikiText-2 raw-v1 benchmark and its scores are not comparable to
  standard word-level perplexity.

The n-gram models use only training counts with add-half smoothing.
`baselines.json` evaluates the same deterministic, non-overlapping target
positions used by a context-16, evaluation-windows-2048 neural run.
Both greedy next-character accuracy and expected sampled reward are recorded,
alongside bits per character. BPC is not word-level perplexity.
