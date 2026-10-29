# Goal
Give me an honest, literature-grounded assessment of this project:
1. Is what we're doing a good idea / worth publishing?
2. How novel is it relative to prior work and current SOTA (incl. 2025–2026 preprints)?
3. What could we add to make it more novel or interesting?

This is a reading-and-research task. Do NOT install the project or run the pipeline.
Treat existing outputs in results/ as the only source of numbers.

# Understand the project first
Read: README.md, AGENTS.md, documents/refreshers/README.md and the latest refreshers it indexes,
documents/prompts/, scripts/classification/README.md, notebooks/classification/README.md.
- Refreshers are dated snapshots: check each against `git log` and the current
  results/{task}/{dataset}/summary.csv before trusting a number.
- Some results are known to be invalid or superseded (e.g. CineMA 2D segmentation bug,
  segmentation re-run under issue #74). Work out which results are current and valid,
  and say explicitly which ones you used and which you excluded and why.
- The project builds on the CineMA paper which is available in this repo under `documents/papers/` and also at this link: https://www.nature.com/articles/s43856-026-01636-0.

# Phases (commit notes after each, under documents/research/)
1. 01_project.md — what the project does: question, models (CineMA, DINOv3, SAM…),
   tasks, datasets (ACDC, M&Ms, M&Ms-2), evaluation protocol, headline numbers.
2. 02_literature.md — the most relevant ~20–30 works: the CineMA paper and work citing it,
   cardiac MRI foundation models, general-vision FMs (DINO/SAM family) applied to medical
   imaging, frozen-backbone/probing benchmarks of FMs. For each: citation + link, what
   it shows, reported numbers on our datasets where available, and relation to us.
   Only cite papers you actually opened. Mark preprints as such.
   Use subagents to search each theme above in parallel, then merge, dedupe and
   verify their results yourself before writing this file.
3. 03_positioning.md — our results vs. published numbers on the same datasets/tasks.
   Note where comparisons aren't like-for-like (splits, metrics, fine-tune vs. probe).
4. REPORT.md — final assessment.

# REPORT.md must contain
- One-paragraph verdict: worthwhile / incremental / already done / unclear, with confidence.
- Closest prior work and exactly how we differ (or don't).
- What's genuinely new in our findings, if anything.
- Strongest argument that this is NOT worth pursuing.
- 3–7 concrete additions that would increase novelty or impact, each with:
  why it's novel (cite the gap), rough effort, and whether our existing code/results support it.
- Open questions you couldn't resolve.

# Rules
- Work on branch `research/novelty-review` from the start.
- Separate claims you verified against a source from claims you inferred.
- If something is blocked (paywall, missing info), write it in documents/research/BLOCKERS.md and continue.
- Be critical, not encouraging. I'd rather hear the idea is weak now than later.
- When done: push branch `research/novelty-review` and open a draft PR summarising REPORT.md.
