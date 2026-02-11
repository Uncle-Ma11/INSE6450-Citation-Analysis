# Citation Analysis: Perfunctory vs. Non-Perfunctory Citation Identification

## Project Goal
The primary objective of this project is to develop an AI system capable of automatically distinguishing between **"Non-Perfunctory Citations"** (Meaningful, Key to the work) and **"Perfunctory Citations"** (Incidental, Background mentions) in scientific literature. 

Identifying both types is equally important for our goal of creating a "weighted" citation analysis metric. We distinguish between mere acknowledgments and citations that truly influenced the methodology or theory of a paper.

## Dataset Documentation

### Primary Dataset: SciCite
We utilize the **SciCite** dataset (AllenAI), utilizing the `isKeyCitation` label as our ground truth for this binary classification.

- **Non-Perfunctory (Meaningful)**: Mapped from `isKeyCitation=True`.
- **Perfunctory (Incidental)**: Mapped from `isKeyCitation=False`.

## Directory Structure
- `data/`: Contains raw datasets (SciCite, ACL-ARC).
- `scripts/`: Python scripts for data processing and modeling.
    - `fetch_abstracts.py`: Retrieves abstracts using Semantic Scholar and OpenAlex APIs.
    - `train_scicite_model.py`: Trains the SciBERT model.
- `docs/`: Project documentation and reports.

## Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```
Key libraries: `torch`, `transformers`, `pandas`, `scikit-learn`, `tqdm`.

### Hardware Setup (RTX 50 Series)
For NVIDIA RTX 50 series GPUs (Blackwell architecture), you must use PyTorch Nightly with **CUDA 12.8** support.

The `requirements.txt` file is configured to pull the nightly build. If you face issues, run:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## How to Run
1. **Data Preparation**:
   The datasets are not included in the repository. Please run the fetcher script to download SciCite, ACT2, and ACL-ARC:
   ```bash
   python scripts/fetch_datasets.py
   ```
   
   Ensure the SciCite dataset is in `data/raw/scicite/`.
   Run the abstract fetcher to enrich the data (optional but recommended):
   ```bash
   python scripts/fetch_abstracts.py
   ```

2. **Training**:
   Train the Single-Head SciBERT model:
   ```bash
   python scripts/train_scicite_model.py
   ```
   Outputs (model checkpoints, plots) will be saved in the root directory.

## Model Logic
- **Input**: `Section Name` + `Citation Context` + `Cited Abstract`
- **Architecture**: `allenai/scibert_scivocab_uncased` + Dropout + LayerNorm + Linear Classifier.
- **Goal**: Maximize Macro F1-Score to handle class imbalance.
