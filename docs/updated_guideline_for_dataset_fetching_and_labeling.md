According to a document from **February 5, 2026**, your target is a model that takes **citation context + cited abstract** and predicts **(a) citation function / “Organic vs. Perfunctory”** and **(b) a numeric “relatedness / intensity” score** between the citing and cited papers. 

Below are the most relevant datasets I found (prioritizing ones that already have *importance / influence* labels, since that’s the closest operationalization of “perfunctory vs not”), followed by a concrete labeling strategy to fill gaps.

---

## Datasets you can use immediately (have “importance” labels)

### 1) ACT2 (Citation Purpose + Citation Influence)

**Why it’s a strong fit:** ACT2 is essentially “your roadmap in dataset form”: citation context + cited abstract (and more) with labels for **purpose** and **influence**.

* **Size / scope:** 4,000 citations, multi-disciplinary.
* **Labels:**

  * **Purpose (6-way):** BACKGROUND, USES, COMPARES_CONTRASTS, MOTIVATION, EXTENSION, FUTURE ([ACL Anthology][1])
  * **Influence (binary):** INCIDENTAL vs INFLUENTIAL ([ACL Anthology][1])
* **Inputs available:** citation context and **cited abstract** are included (plus rich metadata).

**How it maps to your task:**

* **Perfunctory vs not:** start by mapping **INCIDENTAL ≈ perfunctory** and **INFLUENTIAL ≈ organic/meaningful** as a pragmatic first pass.
* **Relatedness score:** treat INFLUENTIAL as “high relatedness” and INCIDENTAL as “low,” then refine to a continuous score using additional signals (see labeling section).

---

### 2) ACT / 3C Shared Task dataset (Citation Purpose + Citation Influence)

If you want a smaller, clean benchmark aligned to shared-task baselines:

* **Size / scope:** 3,000 labeled instances (plus a 1,000-instance held-out test in the shared-task setup). ([wosp.core.ac.uk][2])
* **Same label set as ACT2:** 6-way purpose + binary influence. ([wosp.core.ac.uk][2])
* **Fields:** includes citation context, and the numeric coding for labels is explicitly documented. ([wosp.core.ac.uk][2])

**How to use it:** great for a baseline reproduction + ablation studies before you scale to ACT2/S2ORC.

---

### 3) Valenzuela, Ha, Etzioni (2015) “Identifying Meaningful Citations” dataset

This is *directly* about **citation importance**, and it gives you an **ordinal label set** that is very convenient for a “relatedness/intensity score”.

* **Size:** ~450–465 citations annotated for importance. ([ai2-website.s3.amazonaws.com][3])
* **Fine-grained labels (0–3):**

  * 0 = Related work (incidental)
  * 1 = Comparison (incidental)
  * 2 = Using the work (important)
  * 3 = Extending the work (important) ([ai2-website.s3.amazonaws.com][3])
* **Also provides a collapsed binary mapping** (0/1 incidental vs 2/3 important). ([ai2-website.s3.amazonaws.com][3])
* **Class imbalance note:** only ~14.6% are “important” (2 or 3) in their annotated set. ([ai2-website.s3.amazonaws.com][3])

**How it maps to your task:**

* **Perfunctory vs not:** use the binary collapse as a second “importance” dataset (complements ACT2).
* **Relatedness score:** use 0–3 as **supervised ordinal regression** (and optionally map to 0–1 by dividing by 3).

---

## Datasets that are very useful for citation *function/intent* (but not perfunctory directly)

### 4) SciCite (Citation Intent)

SciCite is widely used for citation intent classification and includes **sectionName** and paper IDs.

* **Size:** train 8,194 / val 916 / test 1,859. ([Hugging Face][4])
* **Labels (3-way intent):** method, background, result. ([Hugging Face][4])
* **Fields helpful for you:**

  * `sectionName` (location signal)
  * `citingPaperId`, `citedPaperId` (lets you fetch metadata/abstracts via Semantic Scholar API)
  * `isKeyCitation` boolean (a weak “importance-ish” signal, though it’s not the same as “influential”). ([Hugging Face][4])

**How to use it:** multi-task training (intent head) alongside ACT2 influence/purpose heads, to improve discourse sensitivity.

---

### 5) ACL-ARC / “Citation Frames” (Jurgens et al.)

This is the classic **citation function** dataset in NLP.

* **Annotated set size:** “nearly 2,000 citations annotated for their function.” ([David Jurgens][5])
* Also includes a **fully-labeled ACL Anthology Reference Corpus** where each citation has an **automatically assigned** function label (good for weak supervision / scaling in-domain). ([David Jurgens][5])
* The ACT shared task documentation notes ACL-ARC is compatible with the ACT schema. ([wosp.core.ac.uk][2])

**How to use it:** domain-specific fine-tuning in NLP + generate silver labels at scale in the ACL domain.

---

### 6) CFC (Citation Function Corpus; Teufel et al.)

Small but fine-grained function labels (useful for taxonomy experiments).

* **Size:** 161 articles; **12 citation function categories**. ([Computer Laboratory][6])
* **License:** CC BY-NC 2.0 (non-commercial). ([Computer Laboratory][6])

---

## Datasets that help learn “relatedness” via evidence alignment

### 7) CL-SciSumm / SciSummNet (citance → cited span mapping)

This is not an “importance” dataset, but it is excellent if you interpret “relatedness” as *how strongly the citation context aligns to specific content in the cited paper*.

* CL-SciSumm provides tasks where **each citance is mapped to referenced text spans in the reference paper** and tagged with an **information facet**. ([GitHub][7])
* The repo describes:

  * a manually annotated set (e.g., 40 document sets for certain tasks),
  * plus a larger auto-annotated “noisy” set (SciSummNet) for training deep models. ([GitHub][7])

**How to use it:** train a model to retrieve the cited spans given the citance; the retrieval confidence can become a component of your relatedness score.

---

## Large corpora to scale up (mostly unlabeled, but essential)

### 8) S2ORC / Semantic Scholar Open Research data

You already mention S2ORC in your roadmap as a source corpus. 【220:0†project_roadmap_draft.md†L11-L16】
The practical update is that **older download methods are out of date** and it has migrated to releases via the **Semantic Scholar API / datasets**. ([GitHub][8])

**How to use it:** mine millions of citation contexts + paper metadata to generate weakly supervised training data (see labeling strategy).

---

## What’s still missing in “ready-made labels”

None of the popular public datasets label *exactly* “perfunctory” in the Moravcsik & Murugesan sense. Your reference paper defines **Organic vs Perfunctory** explicitly (“truly needed for understanding” vs “mainly acknowledgment that other work exists”). 【220:5†What do citation counts measure 1 1.pdf†L33-L37】

So you’ll likely do one of:

1. **Proxy approach:** use **INCIDENTAL/INFLUENTIAL** (ACT/ACT2) as your perfunctory proxy; and/or Valenzuela’s **incidental vs important**. ([ACL Anthology][1])
2. **Relabel / extend:** build a small gold set with your exact taxonomy and then scale via weak supervision.

---

## Labeling strategy that matches your roadmap and the reference paper

Your roaxt window (±3 sentences) and capturing section headers because location correlates with function. 【220:0†project_roadmap_draft.md†L13-L17】 That’s well-supported by the review: low-intensity/perfunctory citations tend to dominate introductions, while high-intensity/meaningful citations appear more in methods/results/discussion. 【220:2†What do citation counts measure 1 1.pdf†L20-L24】

### Step 1: Decide the *annotation unit* and what annotators see

Annotate **each in-text citation mention** (the “#AUTHOR_TAG”-style unit used in ACT). ([wosp.core.ac.uk][2])

For each target citation, show annotators:

* Citation context: **sentence containing the citation + (±_draft.md†L13-L16】
* **Section name / header** (Intro/Methods/Results/Discussion, or fine-grained headings). 【220:0†project_roadmap_draft.md†L16-L17】
* **Cited paper title + abstract** (your model input design). 【220:0†project_roadmap_draft.md†L3-L7】
* Optional but helpful for relatedness: **citing abstract** and/or a short snippet around where the cited paper is discussed multiple times.

### Step 2: Use a 2-layer label scheme (simple + scalable)

You want both a “perfunctory flag” and a “relatedness score”. The cleanest strategy is:

#### Label A: Perfunctory vrom the review* directly:

* **Organic:** citation is truly needed for understanding.
* **Perfunctory:** mainly acknowledgment that related work exists; often no substantive use. 【220:5†What dnnotation heuristics (as guidance, not rules):
* **Perfunctory signals:**

  * citation appears in Intro/Related Work as a list of many citations;
  * “see e.g.” / “for a review” / name-dropping without technical content;
  * no follow-up explanation.
* **Organic signals:**

  * citation supports a key claim, method choice, dataset, equation;
  * appears in Metho†What do citation counts measure 1 1.pdf†L20-L24】

Also explicitly warn annotators about “per citing). The review notes persuasive citations can be common (rees). 【220:10†What do citation counts measure 1 1.pdf†L10-L16】

#### Label B: Relatedness / intensity (ordinal 0–3)

If you want a numeric score, ordinal labels are more reliable than asking humans for a “0.0–1.0” continuous number.

A very usable 0–3 scale is exactly what Valenzuela et al. used:

* 0 = **Related work** (incidental)
* 1 = **Comparison** (incidental)
* 2 = **Using the work** (important)
* 3 = **Extending the work** (important) ([ai2-website.s3.amazonaws.com][3])

Then map to your regression target as: `score = label/3`.

**Why this isoperationalization (“uses/extends” is stronger relatedness than “mentions as related work”), and it’s already a published scheme. ([ai2-website.s3.amazonaws.com][3])

### Step 3: Sampling plan to fight imbalance

Both the review and the importance datasets imply **skew**:

* Perfunctory citations reported anywhere from ~10% to ~50% across studies. 【220:10†What do citation counts measure 1 1.pdf†L5-L16】
* “Important” citat important). ([ai2-website.s3.amazonaws.com][3])

So for your new labels:

* **Stratify sampling by section** (oversample Methods/Results so you actually capture “organic/high intensity” cases).
* **Oversamd multiple times), but note the literature is mixed on whether multiple mentions reliably imply higher importance. 【220:14†What do citation counts measure 1 1.pdf†L15-L19】
* **Ensure multi-discipline coverage** if your end goal is cross-domain robustness (ACT2 already helps here).

### Step 4: Quality control (avoid the classic pitfalls)

The review emphasizes that citation-content analyses can be criticized for subjectivity and methodological inconsistency; experts are “guessing” motivations based on text. 【220:10†What do citation counts measure 1 1.pdf†L41-L45】

So build reliability into the plan:

* **2 annotators per instance** + adjudication on disagreements.
* Track **Cohen’s κ** (binary) and **Krippendorff’s α / weighted κ** (ordinal).
* Maintain an annotation guide with **decision rules** and **counterexamples** (especially for “persuasive” vs “organic”).

### Step 5: Scale up with weak supervision + active learning (when using S2ORC/large corpora)

Once you have a small gold set (even 500–1,000), scale to tens/hundreds of tyle)
Use high-precision heuristics inspired by your roadmap + the review:

* **Section-based LF:** Intro/Background → more likely perfunctory; Methods/Results → more likely organic/high-intensity. 【220:0†project_roadmap_draft.md†L16-L17】【220:2†What do citation counts measure 1 1.pdf†L20-L24】
* **Lexical cue LF:** “we use / we follow / we adopt / based on” → higher intensity; “see e.g. / for a review / survey” → perfunctory.
* **Citation bundle LF:** multiple papers cited together with no elaboration → perfunctory-leaning.
* **Reeness, but treat as *soft* evidence since findings are contradictory. 【220:14†What do citation counts measure 1 1.pdf†L15-L19】

Train a discriminative model on the probabilistic labels, then:

* **Active learning:** repeatedly sample “most uncertain” citations for human labeling to improve decision boundaries fastest.

#### Silver labels via LLM (your roadmap a to label a small subset to form a “silver standard” if human labels aren’t available. 【220:8†project_roadmap_draft.md†L28-L30】

Best practice here: use LLM labels as *bootstrapping*, not as ground truth:

* Keep a **human-verified dev/test set** that the LLM never touches.
* Use consistency checks (same instance paraphrased, order-shuffled evidence) to estimate label noise.

---

## Practical recommendation: the “best starting stack”

If you want a concrete, low-risk path that matches your roadmap:

1. **Start supervised with ACT2** for:

   * perfunctory proxy (INCIDENTAL/INFLUENTIAL),
   * purpose classes for richer structure. ([ACL Anthology][1])

2. **Add Valenzuela (0–3) as the main supervised signal for ([ai2-website.s3.amazonaws.com][3])

3. **Use SciCite as an auxiliary intent task** (background/method/result) and as a source of paper IDs to retrieve metadata/abstracts at scale. ([Hugging Face][4])

4. **Scale with S2ORC-like data via Semantic Scholar** using weak supervision + active learning. ([GitHub][8])

5. If you want “evidence-based relatedness,” incorporate **CL-SciSumm** span alignment as a second to specific content in the cited paper). ([GitHub][7])

---

If you want, I can also propose a unified label mapping so that ACT2 (purpose/influence), Valenzuela (0–3), and your Organic/Perfunctory definitions can be trained in a **single multi-task setup** (classification + ordinal regression) with a consistent evaluation protocol—without needing to redesign your roadmapaft.md†L4-L8】



[1]: https://aclanthology.org/2022.lrec-1.363.pdf "https://aclanthology.org/2022.lrec-1.363.pdf"
[2]: https://wosp.core.ac.uk/jcdl2020/shared_task.html "https://wosp.core.ac.uk/jcdl2020/shared_task.html"
[3]: https://ai2-website.s3.amazonaws.com/publications/ValenzuelaHaMeaningfulCitations.pdf "https://ai2-website.s3.amazonaws.com/publications/ValenzuelaHaMeaningfulCitations.pdf"
[4]: https://huggingface.co/datasets/allenai/scicite "https://huggingface.co/datasets/allenai/scicite"
[5]: https://jurgens.people.si.umich.edu/citation-function/ "https://jurgens.people.si.umich.edu/citation-function/"
[6]: https://www.cl.cam.ac.uk/~sht25/CFC.html "https://www.cl.cam.ac.uk/~sht25/CFC.html"
[7]: https://github.com/WING-NUS/scisumm-corpus "https://github.com/WING-NUS/scisumm-corpus"
[8]: https://github.com/allenai/s2orc/issues/25 "https://github.com/allenai/s2orc/issues/25"
