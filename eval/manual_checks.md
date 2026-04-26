# Manual Pygame Check

Most validation is automated. This is the one human-in-the-loop check needed
before declaring the demo ready. The automated `eval/render_smoke.py` already
covers headless rendering correctness; this doc focuses ONLY on human input
through a live Pygame window (click-drag aiming + click-target selection),
which cannot be reliably scripted.

## How to run

```
cd /path/to/knockout-v2
.venv/bin/python scripts/play.py --opponent heuristic
```

## What to verify

- [ ] Pygame window opens to a 1000x750 square arena (cyan team A, orange team B)
- [ ] Click-and-drag from a Team A penguin: an arrow appears showing aim direction and power
- [ ] Releasing the drag confirms the action; pressing SPACE or ENTER (depending on UX) advances the round
- [ ] All 3 Team A penguins can be aimed in turn before the round resolves
- [ ] After the round, physics plays out (penguins slide, possibly knock each other off the edge)
- [ ] Round ends; new round begins or game-over screen appears
- [ ] No crashes; closing the window cleanly exits

## What to report if something is broken

Paste the exact error traceback or a one-line description of the observed
behavior into `eval/DEBUG_LOG.md` under the `## Manual check failures`
heading. Include which checklist item failed and any reproducer notes
(window size, opponent flag, OS).

## Why we test this manually

Pygame's input event loop and click-target math (penguin selection by mouse
position, drag-vector to action conversion) cannot be reliably automated.
Headless rendering checks confirm that pixels land where expected, but they
cannot confirm that a human's mouse coordinates resolve to the intended
penguin or that drag length maps sensibly onto launch power. A human pair of
eyes catches these misalignments — off-by-N pixel offsets, wrong-team
selections, dead zones near the arena edge — that no automated check sees.
