# Project Roadmap: Citation Analysis Function & Relatedness Model

## Goal
Build a model that takes a **citation context** (the text surrounding a reference) and the **cited abstract** as input to predict:
1.  **Citation Function**: The semantic intent (e.g., Organic vs. Perfunctory, Conceptual vs. Operational).
2.  **Relatedness**: A quantitative score of how essential the cited paper is to the citing paper (Intensity).

## Phase 1: Dataset Acquisition & Preparation
**Objective**: Gather pairs of (Citing Sentence, Cited Abstract) with labels.

### Actionable Steps:
-   **Source Data**: Use an established open-source dataset like **ACL-ARC** (computational linguistics papers) or **S2ORC** (Semantic Scholar Open Research Corpus). These datasets often contain full text and citation links.
-   **Data Extraction (The Context Window)**:
    -   Your reference document notes that context analysis requires reading the text surrounding the citation.
    -   Write a script to extract a "window" (e.g., $\pm 3$ sentences) around the citation marker.
    -   **[refinement] Feature Extraction**: The reference paper highlights that citation *location* (Introduction, Methods, Discussion) correlates strongly with function (e.g., Perfunctory citations are dominant in Introductions; Meaningful ones in Methods/Results). Extract the **Section Header** where the citation appears if possible.
-   **Imbalance Check**:
    -   Be aware that "perfunctory" (superficial) citations can make up a large portion of citations (**25-50%** according to the reference paper).
    -   "Negational" citations are very rare (**~1-2%** to 14%).
    -   *Action*: You will need to balance your dataset (undersampling perfunctory, oversampling negational) or use weighted loss functions so your model doesn't just predict the majority class.

## Phase 2: Defining the Taxonomy (The Labels)
**Objective**: Map your data to the theoretical frameworks provided in your source document.

### Strategy:
Instead of inventing labels, use the validated taxonomies from the literature review.

-   **Recommended Classification Scheme**:
    -   **Binary Task (Fundamental Split)**:
        -   **Organic** (Essential/High Intensity): Needed for understanding the paper.
        -   **Perfunctory** (Peripheral/Low Intensity): Acknowledgment only, often in introductory lists.
    -   **Multi-Class Task (Advanced)**:
        -   *Option A (Moravcsik & Murugesan, 1975)*:
            -   **Conceptual vs. Operational**: Theory vs. Tool/Method.
            -   **Evolutionary vs. Juxtapositional**: Building on vs. Alternative to.
            -   **Confirmative vs. Negational**: Correct vs. Disputed.
        -   *Option B (Refined Unified Typology - Bornmann & Daniel, 2008)*:
            -   **Affirmational**: Confirms/supports the cited work.
            -   **Assumptive**: General background/history.
            -   **Conceptual**: Deﬁnitions, concepts, theories.
            -   **Methodological**: Tools, techniques, design.
            -   **Contrastive**: Contrasts current work with cited work.
            -   **Negational**: Disputes/corrects cited work.
            -   **Persuasive**: Ceremonial/Authority citing.

## Phase 3: Model Development (Switching from LLM)
**Objective**: Train a specialized model that outperforms a generic LLM prompt.

### Baseline (The LLM):
-   Continue using your current LLM agent to label a small subset (e.g., 500 citations) to create a "Silver Standard" dataset if human labels aren't available.

### The Upgrade (The Student Model):
-   **Architecture**: Fine-tune **SciBERT** (or standard BERT). SciBERT is pre-trained on scientific text, making it better at understanding the "jargon" and "language barriers" mentioned in texts.
-   **Input Formatting**:
    $$ \text{[CLS]} \text{ Citation Context } \text{[SEP]} \text{ Cited Abstract } \text{[SEP]} \text{ (Optional: Section Name) } $$
-   **Tasks**:
    -   **Head A (Classification)**: A standard Softmax layer to predict the class (e.g., Evolutionary, Operational, Perfunctory).
    -   **Head B (Regression)**: A linear layer to predict a **Relatedness Score** (0-1), addressing the issue that simple citation counts fail to measure the *intensity* of influence.

## Phase 4: Evaluation & Analysis
**Objective**: Prove your model works using standard metrics.

### Metrics:
-   **Macro F1-Score**: Crucial because of the extreme class imbalance (Negational is rare).
-   **Per-Class Precision/Recall**: Specifically monitor the "Negational" class performance, as it is scientifically interesting but hard to catch.
-   **Confusion Matrix**: Analyze if "Perfunctory" citations are being confused with "General Background" (Assumptive).

### The "Why This Matters" Analysis:
-   Compare your model's predictions against raw citation counts.
-   Your document argues that citation counts are flawed because they include "redundant" or "perfunctory" references.
-   **Success Criteria**: Show that your model can filter out these "junk" citations to create a **"Weighted Citation Score"** that better reflects true scientific impact.

## Suggestions for Project Refinement
1.  **Leverage Citation Location**: The reference paper strongly suggests that *where* a citation appears is a strong signal of its intent. If your data extraction pipeline can capture "Introduction" vs "Methodology", feed this into the model (either as text or a categorical feature).
2.  **Focus on "Intensity"**: The distinction between "Cursory/Low Intensity" and "Meaningful/High Intensity" (Maricic et al., 1998) might be more practical and reliable than complex semantic labels. Consider making indices (Regression Head) a priority.
3.  **Handling "Persuasion"**: The social constructivist view suggests citations are used for persuasion (citing authorities). These might look like "Organic" citations but serve a "Perfunctory" (social) purpose. This is a hard edge case to consider during error analysis.