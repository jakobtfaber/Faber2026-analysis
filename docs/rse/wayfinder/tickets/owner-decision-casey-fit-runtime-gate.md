# Casey joint fit: runtime requirement vs exact likelihood

- Type: `wayfinder:task` (HITL)
- Status: resolved (owner ruling 2026-08-04 superseded the three-way choice)
- Assignee: Owner
- Blocked by: none
- Map: [ApJ submission](../map-apj-submission.md)

## Resolution (2026-08-04)

The owner resolved this by a fourth path: ratifying band-specific compact
fit-input windows (CHIME `[153:221]`, DSA `[1300:1323]`, both arms) after
visual vetting in the input-diagnostic notebook. Redefining the delivered
inputs dissolves the likelihood-equivalence impasse; the measured ~2 ms /
~8 ms per-call timings become the legitimate fit cost (~31 min projection
for all four fits at measured pool efficiency). Decision record:
`~/Data/Faber2026/review/casey-joint-fit-inputs/window-decision-ratified-20260804.json`.
Sampler restart remains separately gated.

## Owner decision card (superseded)

The Casey performance-recovery circle (Codex advisor + three researchers, plus
an independent Claude adversarial review at
`~/Data/Faber2026/review/casey-fit-performance-recovery/adversarial-review-claude-20260804.md`)
has exhausted the window-truncation candidates. Every compact-window variant
fails the 1e-8 likelihood-equivalence gate, as expected physically: the sampler
explores prior edges where the scattering tail fills any window. The exact
(likelihood-preserving) optimizations achieved so far reduce per-call cost to
37.5 ms (Gaussian) / 100 ms (scattering), projecting ~419 minutes for all four
fits after measured pool efficiency — roughly 7 hours, versus the standing
owner requirement of minutes-scale. The circle is paused; nothing committed,
no checkpoints touched, no sampler running.

```json
{
  "id": "owner-decision-casey-fit-runtime-gate",
  "kind": "scientific",
  "title": "Casey fit: relax the minutes-scale runtime gate or pursue compiled kernels",
  "decision": "The exact likelihood cannot currently meet the minutes-scale runtime requirement. Choose the path forward.",
  "recommended": {
    "choice": "compiled-kernel-first",
    "reason": "A compiled/parallel exp-erfcx kernel path is exact math (no likelihood change), plausibly recovers ~10x, and preserves the runtime requirement; the ~7-hour exact run remains the fallback if it falls short."
  },
  "choices": [
    {
      "id": "compiled-kernel-first",
      "label": "Authorize an implementation-level compiled kernel attempt (numba/C/threads) under the same 1e-8 equivalence gate before revisiting the runtime requirement."
    },
    {
      "id": "accept-hours",
      "label": "Relax the minutes-scale requirement and authorize the ~7-hour exact full-prior run on h17 with the current verified candidate."
    },
    {
      "id": "hold",
      "label": "Hold the lane paused; no further performance work."
    }
  ]
}
```
