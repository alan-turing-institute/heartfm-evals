# refreshers/

Point-in-time snapshots of where the project stands, written when someone returns to the
codebase after a break. Each file answers "what is the state of task X right now, which branch
holds the real results, what are the numbers, and what is known to be broken?" — the things that
are expensive to reconstruct from `git log` and a directory of JSONs.

Naming: `YYYY-MM-DD-{task}.md`.

**These are snapshots, not living documents.** They are accurate as of their date and the commit
SHAs recorded in their header, and they are not updated when the code moves. Before trusting any
number or claim in one, check its date against `git log` and re-read the current
`results/{task}/{dataset}/summary.csv`. When the picture changes materially, write a new
snapshot rather than editing an old one.

For durable design rationale (why a decision was made), see [../prompts/](../prompts/) instead —
that is the decision-record folder and it *is* meant to persist.

## Index

- [2026-10-22-segmentation.md](2026-10-22-segmentation.md) — **current for segmentation.**
  Committed results deleted ahead of the ACDC re-run (issue #74), nothing regenerated yet,
  and the DINOv3-S 2D ACDC cache holds a second train/val split that must be repaired first.
- [2026-09-11-segmentation.md](2026-09-11-segmentation.md) — *superseded by the above.*
  SAM2 removed and archived, the SAM v1 off-by-one fixed (taps moved to the
  global-attention blocks), tests/CI restored — and the two blockers that remain: disk and
  SAM extraction time.
- [2026-09-01-segmentation.md](2026-09-01-segmentation.md) — *superseded by the above.*
  The CineMA 2D z-misalignment bug (invalidates all committed CineMA Linear/Conv numbers),
  SAM v1 arriving on the mainline, and the disk/compute blockers for the re-run.
- [2026-08-09-segmentation.md](2026-08-09-segmentation.md) — *superseded by the above.* Segmentation code path, the
  `main` / `sam2-classification` fork, full Dice tables, and eight known issues.
- [2026-10-29-classification.md](2026-10-29-classification.md) — **current for classification.**
  `finetune` renamed to `--head linear`, dead `ft-full` code and the duplicate M&Ms JSON gone;
  committed summaries not regenerated since; two conflicting published-CineMA baselines; issue
  #82 (CineMA-paper-style head) as the next real work.
- [2026-10-23-classification.md](2026-10-23-classification.md) — *superseded by the above.*
  No result has changed since 08-09; one of eleven issues fixed, the SAM v1 fix shown not to
  affect classification, and `sam2-classification` now needing a port rather than a merge.
- [2026-08-09-classification.md](2026-08-09-classification.md) — *superseded by the above, but
  still the reference for the numbers.* Classification code path,
  AUC/accuracy tables across all three datasets, the gap to the published CineMA fine-tune,
  and eleven known issues.
