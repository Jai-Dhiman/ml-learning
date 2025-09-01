# 30-Day Twitter Content Calendar — Piano Analysis, Dataset Recreation, and JAX/Flax Learning

Goal: Build an audience by recreating a PercePiano‑style dataset from Chopin/Liszt recordings while learning and sharing ML + JAX/Flax. Cadence targets: 1 original post/day, 2 threads/week, and 5–10 meaningful replies/day.

Posting windows (start here, then A/B test):

- Weekdays: 8–10am, 12–2pm, or 6–9pm local
- Weekends: 9–11am or 6–8pm local
- Pick one window and stick with it for a week, then compare engagement and double down on the best slot.

Daily engagement checklist:

- [ ] Leave 5–10 substantive replies (from curated Lists: Audio ML/MIR, JAX/Flax, Music Tech Builders)
- [ ] Add alt text to all visuals
- [ ] Ask 1 specific question per post to invite replies
- [ ] Track replies, bookmarks, follows; log learnings

Working dimension list (inspired by PercePiano):

1) Timing precision (onset accuracy)
2) Microtiming / rubato
3) Articulation (staccato–legato)
4) Pedal usage
5) Dynamics (loudness)
6) Dynamic range / contrast
7) Timbre brightness (spectral centroid)
8) Timbre warmth / richness
9) Voicing / balance (melody vs accompaniment)
10) Phrasing / segmentation
11) Accentuation / attack
12) Tempo stability
13) Sustain / decay control
14) Clarity (noise / blur)
15) Technical precision (errors)
16) Expressiveness / musicality
17) Emotional intensity / valence
18) Interpretation style (romantic vs classical)
19) Overall performance quality

How to use this calendar:

- Each day lists: Theme, Post Type, Asset, Hook, CTA, Suggested Time
- Keep 0–2 relevant hashtags (e.g., #AudioML #JAX #MIR #Piano)
- If a day is too heavy, swap the post type to a single-image update

---

## Week 1 (Days 1–7): Kickoff, dataset plan, first baselines

Day 1 — Pinned Starter Thread

- Post Type: 5-tweet thread (pin it)
- Asset: Header image + optional 10–20s demo
- Hook: “Recreating a PercePiano‑style dataset with Chopin/Liszt while learning ML + JAX/Flax.”
- CTA: “Follow for daily logs and weekly lessons. Repo link.”
- Time: 8–10am

Day 2 — Dataset Plan (Chopin/Liszt)

- Post Type: 1–2 tweets
- Asset: Diagram: sourcing → slicing (10–30s) → labeling → metadata
- Hook: “Public‑domain Chopin/Liszt → 10–30s excerpts → 19‑dim ratings. Open‑sourcing preprocessing + labeling.”
- CTA: Ask for public‑domain source pointers and licensing gotchas
- Time: 12–2pm

Day 3 — Label Schema v0.1 + Rater Calibration

- Post Type: 1 tweet + image
- Asset: Image of 19‑dim schema; short rubric examples
- Hook: “Rebuilding the 19‑dimension rating schema + calibration plan to align raters.”
- CTA: Ask for feedback on rubric clarity
- Time: 6–8pm

Day 4 — Audio Pipeline Baseline

- Post Type: Short micro‑thread (2–3 tweets)
- Asset: Spectrogram with config (sample rate, n_fft, hop, n_mels)
- Hook: “Baseline mel pipeline; how I avoid leakage and over‑augmentation.”
- CTA: Ask for preferred mel configs from MIR folks
- Time: 8–10am

Day 5 — JAX/Flax Lesson #1

- Post Type: 1 tweet + code image (or gist link)
- Asset: Minimal TrainState + pure train_step pattern
- Hook: “TrainState + jit basics that keep experiments reproducible.”
- CTA: Invite beginners to ask questions; promise runnable snippet soon
- Time: 12–2pm

Day 6 — First Experiment: Timing Baseline

- Post Type: 1 tweet
- Asset: Prediction vs GT overlay (line chart or video)
- Hook: “Timing baseline: where it works and where rubato breaks it.”
- CTA: Ask for piece suggestions to stress‑test
- Time: 6–8pm

Day 7 — Weekly Recap Thread

- Post Type: Thread (5–8 tweets)
- Asset: 2–3 visuals (pipeline, schema snippet, result overlay)
- Hook: “What shipped, what broke, what’s next.”
- CTA: Ask for 1 thing to prioritize next week
- Time: 9–11am (weekend)

---

## Week 2 (Days 8–14): Start “19 Days of PercePiano” + infra thread

Day 8 — Dimension 1: Timing Precision

- Post Type: Single post
- Asset: Short audio clip + overlay
- Hook: “Why onset accuracy matters for musical feel.”
- CTA: Ask for challenging examples
- Time: 8–10am

Day 9 — Dimension 2: Microtiming / Rubato

- Post Type: Single post
- Asset: A/B clip (strict vs rubato passage)
- Hook: “Where models confuse expressiveness for error.”
- CTA: Ask for composer passages with rubato extremes
- Time: 12–2pm

Day 10 — Dimension 3: Articulation (Staccato–Legato)

- Post Type: Single post
- Asset: Spectrogram zooms of note tails
- Hook: “Detecting articulation beyond note on/offs.”
- CTA: Ask for articulation‑rich excerpts
- Time: 6–8pm

Day 11 — Dimension 4: Pedal Usage

- Post Type: Single post
- Asset: Visual of decay envelopes
- Hook: “Pedal detection via decay and resonance cues.”
- CTA: Ask for pedal‑heavy performances
- Time: 8–10am

Day 12 — Dimension 5: Dynamics

- Post Type: Single post
- Asset: Loudness curve overlay
- Hook: “Mapping dynamics to perceived intensity.”
- CTA: Ask for dynamic contrast examples
- Time: 12–2pm

Day 13 — Dimension 6: Dynamic Range / Contrast

- Post Type: Single post
- Asset: Histogram or range visualization
- Hook: “How contrast shapes phrasing impact.”
- CTA: Ask for big‑range recordings
- Time: 6–8pm

Day 14 — Thread: Dataset Sourcing + Rater Calibration

- Post Type: Thread (5–8 tweets)
- Asset: Flowchart; mini rubric snippet
- Hook: “How I source Chopin/Liszt and keep rater reliability sane.”
- CTA: Ask for raters or rubric critiques
- Time: 9–11am (weekend)

---

## Week 3 (Days 15–21): Continue dimensions + model deep dive

Day 15 — Dimension 7: Timbre Brightness

- Post Type: Single post
- Asset: Spectral centroid visuals
- Hook: “Centroid correlates with ‘brightness’—but not the whole story.”
- CTA: Ask for pieces with brightness shifts
- Time: 8–10am

Day 16 — Dimension 8: Timbre Warmth / Richness

- Post Type: Single post
- Asset: Rolloff/flatness comparison
- Hook: “Warmth beyond EQ curves—context matters.”
- CTA: Ask for warm vs bright comparisons
- Time: 12–2pm

Day 17 — Dimension 9: Voicing / Balance

- Post Type: Single post
- Asset: Melody vs accompaniment energy
- Hook: “Can models track melody prominence?”
- CTA: Ask for voicing‑rich passages
- Time: 6–8pm

Day 18 — Dimension 10: Phrasing / Segmentation

- Post Type: Single post
- Asset: Phrase boundary visualization
- Hook: “Detecting musical sentences in audio.”
- CTA: Ask for phrasing exemplars
- Time: 8–10am

Day 19 — Dimension 11: Accentuation / Attack

- Post Type: Single post
- Asset: Onset strength curves
- Hook: “Attack profiles as expression cues.”
- CTA: Ask for accented dance forms
- Time: 12–2pm

Day 20 — Dimension 12: Tempo Stability

- Post Type: Single post
- Asset: Tempo estimate over time
- Hook: “Stability vs expressivity tradeoffs.”
- CTA: Ask for metronomic vs flexible performances
- Time: 6–8pm

Day 21 — Thread: Model Architecture Deep Dive

- Post Type: Thread (6–10 tweets)
- Asset: CNN diagram + multi‑task heads
- Hook: “Why this backbone and how heads decouple dimensions.”
- CTA: Ask for architecture suggestions
- Time: 9–11am (weekend)

---

## Week 4 (Days 22–30): Finish dimensions, interpretability, MLOps, demos

Day 22 — Dimension 13: Sustain / Decay Control

- Post Type: Single post
- Asset: Envelope shapes across notes
- Hook: “Decay as a performance fingerprint.”
- CTA: Ask for legato pedaling examples
- Time: 8–10am

Day 23 — Dimension 14: Clarity

- Post Type: Single post
- Asset: SNR/blur proxy visualization
- Hook: “Clarity vs room color and recording noise.”
- CTA: Ask for clean vs noisy takes
- Time: 12–2pm

Day 24 — Dimension 15: Technical Precision

- Post Type: Single post
- Asset: Error spikes or artifacts
- Hook: “Where models misread virtuosity as error.”
- CTA: Ask for etudes with tricky passages
- Time: 6–8pm

Day 25 — Dimension 16: Expressiveness / Musicality

- Post Type: Single post
- Asset: Multi‑feature overlay
- Hook: “Quantifying ‘feel’ (imperfectly).”
- CTA: Ask for subjective takes
- Time: 8–10am

Day 26 — Dimension 17: Emotional Intensity / Valence

- Post Type: Single post
- Asset: A/B emotional contrast
- Hook: “Can audio features hint at emotion ratings?”
- CTA: Ask for emotionally charged excerpts
- Time: 12–2pm

Day 27 — Dimension 18: Interpretation Style

- Post Type: Single post
- Asset: Style spectrum (romantic ↔ classical)
- Hook: “Style cues models latch onto.”
- CTA: Ask for performer comparisons
- Time: 6–8pm

Day 28 — Thread: MLOps (W&B, Checkpoints, GKE)

- Post Type: Thread (5–8 tweets)
- Asset: W&B screenshots; run diagram
- Hook: “What actually made training sane and reproducible.”
- CTA: Ask for infra tips
- Time: 9–11am (weekend)

Day 29 — Dimension 19: Overall Performance Quality + Real‑Time Demo

- Post Type: Short video demo
- Asset: 10–20s inference clip overlaying multiple heads
- Hook: “Real‑time‑ish predictions; where it shines and fails.”
- CTA: Ask what to build next (feature requests)
- Time: 8–10am

Day 30 — Final Recap Thread + What’s Next

- Post Type: Thread (8–12 tweets)
- Asset: Best visuals; small metrics summary
- Hook: “30 days of piano analysis: what I learned building the dataset and models in public.”
- CTA: Invite collaborators/raters; encourage follows for the next phase (multi‑spectral fusion / deployment)
- Time: 12–2pm

---

Weekly learning posts (sprinkle into threads or standalone):

- JAX/Flax #1: TrainState + pure train_step (Week 1, Day 5)
- JAX/Flax #2: jit/vmap patterns for audio batches (Week 2)
- Multi‑task #1: Loss weighting and head calibration (Week 3)
- Eval #1: Spearman vs MAE, reliability, and what “good” looks like (Week 3)
- Interpretability #1: Grad‑CAM/attention on spectrograms (Week 4)

Tracking template (copy per week):

- Posts published: [ ] [ ] [ ] [ ] [ ] [ ] [ ]
- Threads published: [ ] [ ]
- Avg replies per post: __| Bookmarks:__  | New follows: __
- Top time slot: __
- What to double down on next week: __

Notes:

- Keep links in a reply when possible; avoid hashtag soup; tag people only when truly relevant
- Always include alt text on images and concise captions on videos
- If you miss a day, don’t “make up” the content—just continue the plan and keep quality high
