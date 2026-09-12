# Token-synchronized anatomical brain activity

The offline renderer places every recorded model state at its measured MaleCNS
soma coordinate and advances one frame per generated BPE token. It consumes
recovered activation artifacts, so rendering does not retrain the model or
allocate a GPU.

Three views are available:

- `activity_3d`: the default fixed-camera 3D view for comparing token-linked
  changes without camera motion;
- `projections`: fixed X-Y and X-Z projections for easier spatial comparison.
- `rotating_3d`: an anatomical overview whose moving camera should not be used
  to compare activity between adjacent tokens.

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
  --view activity_3d \
  --frames-per-second 3
```

Use `--view projections` for the fixed two-panel version or `--view rotating_3d`
for an anatomical rotation. GIF and MP4 outputs are selected by the output
filename extension. MP4 rendering requires `ffmpeg`.

## Reading the animation

- Each frame is the recurrent state used to select the displayed token.
- The displayed token has not yet been fed back into the circuit.
- Blue and red encode negative and positive change in continuous model state
  since the preceding token-selection frame.
- Brightness and point size encode absolute state change since the previous
  token selection, scaled once across the full recording. The first frame uses
  the zero state as its baseline because no previous recorded frame exists.
- The text line is decoded from the complete token prefix, not by concatenating
  independently decoded byte-level tokens.

These values are changes between consecutive model states at anatomical soma
positions. They are not measured biological firing rates, attention weights,
inhibitory identities or causal importance. A frame is associated with the
displayed token selection; it does not show that the displayed token caused the
change. The animation does not show a living fly producing language.
