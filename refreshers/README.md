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

- [2026-08-09-segmentation.md](2026-08-09-segmentation.md) — segmentation code path, the
  `main` / `sam2-classification` fork, full Dice tables, and eight known issues.
