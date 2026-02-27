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
It is highly recommended to use a virtual environment. Activate it using:
```bash
.\venv\Scripts\Activate
```
Then install required packages using:
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

2. **Model Training & Efficiency Monitoring**:
   Train the Single-Head SciBERT model. The script is configured to automatically calculate and print **Training Efficiency Metrics**, including Total Training Time, Time per Epoch, Peak VRAM usage, and estimated FLOPS using the `thop` package.
   ```bash
   python scripts/train_scicite_model.py
   ```
   *   **Metrics Evaluation**: The script tracks training loss, validation accuracy, and Macro F1-Score (to correctly evaluate the class imbalance).
   *   **Outputs**: The best model weights will be saved to `best_model.pt` and a learning curve graph will be plotted to `results/learning curve.png`.

3. **Inference Benchmarking & Quantization**:
   To test the deployment viability and measure exact inference metrics on your hardware, run the benchmark script:
   ```bash
   python scripts/benchmark_inference.py
   ```
   This script performs the following evaluations:
   *   **Inference Efficiency**: Measures Latency (p50 and p90) on a batch size of 1, Throughput (samples/sec) on a batch size of 32, and Peak VRAM footprint.
   *   **Model Quantization**: The script automatically applies **PyTorch Dynamic Quantization**, converting the model's FP32 `Linear` layers to `INT8`. It then runs the CPU inference benchmark again on the quantized model to compare the latency speedups, model file size reduction, and evaluates the exact `Accuracy` and `F1 Score` degradation against the original model.

4. **Single-sample Inference**:
   To test the model on a single, custom citation example, use the inference script:
   ```bash
   python scripts/inference_scicite.py \
       --context "This method improves upon previous work by using a transformer." \
       --section "Methods" \
       --abstract "Optional cited paper abstract text here."
   ```

## Model Logic
- **Input Representation**: `Section Name` + `Citation Context` + `Cited Abstract`
- **Architecture**: `allenai/scibert_scivocab_uncased` + Dropout + LayerNorm + Linear Classifier.
- **Goal**: Maximize Macro F1-Score to accurately weigh both Perfunctory and Non-Perfunctory classes.
