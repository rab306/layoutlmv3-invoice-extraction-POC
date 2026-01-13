# Invoice Entity Extraction POC - LayoutLMv3

## Project Overview

This is a **personal proof-of-concept (POC) project** exploring invoice entity extraction from **real-world scanned invoices** using **LayoutLMv3-Base**.  
The project emphasizes **engineering solutions to practical challenges** rather than chasing benchmark scores.  

Key focus areas:
- Handling **messy OCR outputs** with errors
- Extracting **multi-word entities** (addresses, product descriptions)
- Processing **tables and numeric fields**
- Building a **production-like pipeline** for data preprocessing, label alignment, and HuggingFace-ready dataset creation
- Showcasing **systematic ML engineering practices** (cascading strategies, spatial heuristics, threshold tuning, precision/recall balance)

---

## Project Scope

- **Personal POC project** experimenting with real-world invoice challenges  
- **Pipeline**: Raw OCR → Label alignment → HuggingFace dataset → Model training → Evaluation  
- **Engineering focus**: Multi-strategy matching, spatial heuristics, numeric field handling  
- **Goal**: Learn, prototype, and build reproducible workflows, not to achieve state-of-the-art metrics

---

## Dataset

* **Source:** HuggingFace → `Aswin-M/7000_invoice_image_and_json_1`
* **Samples:** 6,937 invoices
* **Tokens:** 268,934
* **Entity tokens:** 50,285 (18.7% coverage)
* **Data type:** Images (PNG) + aligned JSON labels
* **Split:** Train/Val/Test → 80/10/10

---

## Phases Overview

| Phase       | Objective                                | Key Contributions                                                                                                                                                                                                               |
| ----------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Data exploration & OCR preprocessing     | - Examined JSON null patterns<br>- Verified image quality<br>- Decided lightweight OCR sufficient                                                                                                                               |
| **Phase 2** | Baseline entity alignment                | - Exact + fuzzy string matching<br>- Initial labeling (~13% coverage)<br>- Highlighted numeric field challenges                                                                                                                 |
| **Phase 3** | Advanced matching & heuristics           | - Multi-strategy numeric matching (4-strategy)<br>- Address sequence & spatial matching<br>- Product table row grouping<br>- Threshold tuning (4 rounds)<br>- Entity coverage: 18.7%                                            |
| **Phase 4** | Label statistics & analysis              | - Field-by-field coverage tiers<br>- Precision/Recall trade-off analysis<br>- Identified OCR as main bottleneck                                                                                                                 |
| **Phase 5** | Dataset creation (LayoutLMv3)            | - Preprocessing images + tokens + bounding boxes<br>- Memory-safe chunked processing<br>- Train/Val/Test splits (HF Datasets, Arrow format)<br>- Final size: 4.17 GB                                                            |
| **Phase 6** | Model fine-tuning                        | - HuggingFace Trainer with weighted loss<br>- Phased experiments: timid → aggressive → balanced<br>- Precision/Recall gap <10%<br>- Global F1: ~54.68%                                                                          |
| **Phase 7** | Final refinements | - Square root smoothing for class weights<br>- Numeric & multi-word field post-processing<br>- Stable inference pipeline<br>- Final model robust to OCR noise<br>- Systematic threshold documentation for production deployment |

---

## Pipeline Summary

1. **OCR & Preprocessing**

   * Light-weight OCR for clean images
   * Word extraction + bounding boxes
   * Null handling in JSON labels

2. **Label Alignment**

   * Cascading matching strategies (exact → substring → fuzzy → lenient)
   * Field-specific logic: numeric, addresses, products
   * Spatial heuristics to prevent cross-row errors

3. **Dataset Creation**

   * LayoutLMv3Processor: image, token, bbox alignment
   * Chunked processing to avoid memory crashes
   * Train/Val/Test: 80/10/10
   * Arrow format for memory-mapped training

4. **Model Training**

   * HuggingFace Trainer
   * Weighted loss: timid → aggressive → balanced
   * Precision/Recall monitoring
   * Square root smoothing for rare classes

5. **Inference & Post-processing**

   * Token-level to word-level label alignment
   * Numeric field verification & correction
   * Multi-word entity reconstruction
   * Thresholds tuned for production reliability

---

## Key Insights & Engineering Decisions

* **Numbers ≠ Text:** Custom numeric matching increased coverage 2–3×
* **Spatial heuristics matter:** Y-coordinate & distance grouping reduced false positives ~50%
* **Chunked dataset processing:** Enabled 4 GB dataset creation with 1.5 GB peak memory
* **Threshold tuning:** Systematic 4-round optimization, avoiding guesswork
* **Precision > Recall in noisy OCR:** High-quality labels prioritized over total coverage
* **Phased model training:** Progressive weighting strategy stabilized F1, reduced hallucinations

---

## Folder Structure

```
invoiceOCR/
├── images/                    # Raw invoice images
├── labeled_dataset.json        # Phase 5 output
├── layoutlm_dataset/           # Phase 6 preprocessed HF Dataset
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── preprocessing.py
│   ├── matching.py
│   ├── dataset_creation.py
│   └── train_model.py
├── notebooks/
├── tests/
├── results/
├── requirements.txt
└── README.md
```

---

## Quick Start

1. **Clone Dataset**

```bash
git clone https://huggingface.co/Aswin-M/7000_invoice_image_and_json_1
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Preprocess & Align Labels**

```bash
python src/preprocessing.py
```

4. **Create HuggingFace Dataset**

```bash
python src/dataset_creation.py
```

5. **Train Model**

```bash
python src/train_model.py
```

6. **Evaluate / Inference**

```bash
python src/train_model.py --evaluate
```

---

## Final Remarks

* **Total Labels:** 50,285 entities
* **Entity Coverage:** 18.7% 
* **Peak Memory During Dataset Creation:** 1.5 GB
* **Training Stability:** F1 ~54.68%, Precision-Recall gap <10%
* **Pipeline:** Fully reproducible, robust to noisy OCR, ready for new invoice data