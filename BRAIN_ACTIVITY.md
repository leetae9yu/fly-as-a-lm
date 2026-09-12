# Token-synchronized anatomical brain activity

The offline renderer places every recorded model state at its measured MaleCNS
soma coordinate and advances one frame per generated BPE token. It consumes
recovered activation artifacts, so rendering does not retrain the model or
allocate a GPU.

Two views are available:

- `rotating_3d`: a rotating 3D soma cloud suitable for video and README previews;
- `projections`: fixed X-Y and X-Z projections for easier spatial comparison.

For the central N5,600 replication graph, 5,576 of 5,600 neurons have a measured
`somaLocation`. The remaining 24 are omitted rather than assigned invented
positions. Native source coordinates are displayed without claiming an
unverified physical unit.

## Render the published seed-1 recording

Install the language and connectome extras, then render a shareable MP4. If
`data/large_connectome/raw/annotations.feather` is absent, first run
`python -m scripts.prepare_large_connectome` to download and verify the pinned
MaleCNS source data.

```bash
uv sync --extra language --extra connectome
flyrl-brain-activity \
  --activation-dir results/central-brain-replication-t4-final/seed-1/real \
  --annotations data/large_connectome/raw/annotations.feather \
  --tokenizer artifacts/tinystories-t4-corpus/tokenizer.json \
  --output brain-activity-3d.mp4 \
  --view rotating_3d \
  --frames-per-second 3
```

Use `--view projections` for the fixed two-panel version. GIF and MP4 outputs
are selected by the output filename extension. MP4 rendering requires `ffmpeg`.

## Reading the animation

- Each frame is the recurrent state used to select the displayed token.
- The displayed token has not yet been fed back into the circuit.
- Color encodes the signed continuous model state on a fixed negative-to-positive
  scale.
- Brightness and point size encode absolute state change since the previous
  token selection, scaled once across the full recording. The first frame uses
  the zero state as its baseline because no previous recorded frame exists.
- The text line is decoded from the complete token prefix, not by concatenating
  independently decoded byte-level tokens.

These values are model states at anatomical soma positions. They are not
measured biological firing rates, attention weights, inhibitory identities or
causal importance. The animation shows where the engineered recurrent model
changes while generating text; it does not show a living fly producing
language.
