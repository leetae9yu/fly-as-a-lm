# Partial BPE publication audit

Scope: publish existing six completed runs and stopped-study limitations.
No new training, profiling, GPU allocation or training-containing tests.

| Scenario | Check | Observed evidence | Verdict |
| --- | --- | --- | --- |
| Completed run coverage | Read six saved reports | Seed 0 five conditions and seed 1 real, each 8,000 updates and trace 1..8000 | Pass |
| Paired inputs | Compare saved corpus/graph IDs and seed 0 checkpoint RNG | Same pinned hashes; exact window RNG across five seed 0 checkpoints | Pass |
| Checkpoint recovery | Hash six local checkpoints against verification JSON | All six source hashes match; restoration parameter/Adam differences 0, RNG/progress/trace true | Pass |
| Generation reproduction | Read verification JSON | stored_generation_reproduced=true for all six | Pass |
| Numerical continuation | Read six continuation records | Max parameter difference 2.384185791015625e-7; max Adam difference 3.026798367500305e-9; loss differences 0 | Pass |
| Missing runs | Compare plan with recovered artifacts | Six completed, one 4,250-update external backup, eight not run; no pooled three-seed claim | Pass |
| Scores and units | Read final metrics and check log(perplexity)=NLL | 1024 target windows; BPE BPC=null; README/report use test scores, validation separately labeled | Pass |
| Bottleneck scope | Read CPU diagnostic JSON | Device CPU, fresh model, 3 warm timed updates; GPU kernel attribution explicitly unavailable | Pass |
| Documentation links | Resolve 54 relative links across five documents | No missing local targets | Pass |
| Packaging | uv build --no-sources | Wheel and sdist built, exit 0; no experiment launched | Pass |
| Whitespace | git diff --check | Exit 0 | Pass |
| Authorship | git config user.name/user.email | leetae9yu / 192941348+leetae9yu@users.noreply.github.com | Pass |

Primary evidence:
- results/bpe-main/seed-0/*/report.json
- results/bpe-main/seed-1/real/report.json
- results/bpe-resume/seed-0/*/verification.json
- results/bpe-resume/seed-1/real/verification.json
- results/bpe-bottleneck-cpu/real.json
- results/bpe-runtime-interruption.json

Existing implementation validation, not rerun for this publication:
192 passed / 1 CUDA-only skip locally; 48 model tests passed on T4; 3 subsequent
study/CLI tests passed. User explicitly stopped further experiments.
