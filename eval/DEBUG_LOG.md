# Audit Debug Log

Working doc for the audit pass on knockout-v2. Each entry records a failure
discovered by the harness, its diagnosis, and the resolution. Distilled into
RESEARCH_NOTES.md at audit completion; this file is then deleted.

## Format

```
### [bot name or component]: [one-line summary]

- **Discovered by**: smoke.py / pairwise_smoke.py / mini_tournament.py / pytest / render_smoke.py
- **Failure mode**: <one or two sentences>
- **Bucket**: A=stale-path | B=dim-mismatch | C=code-regression | D1=loads-but-weak (acceptable) | D2=action-probe-fails (broken) | E=other
- **Effort**: trivial | small | medium | large
- **Resolution**: <fixed at commit SHA / dropped from lineup / future work>
```

## Open

(Add new entries here when discovered.)

## Resolved

(Move entries here once fixed, with the resolution filled in.)

## Manual check failures

(Populated only if the user's manual Pygame check surfaces issues.)
