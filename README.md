# 🏥 HEALBOT: Medical Intelligent Diagnosis and Q&A System

> An AI-powered medical assistant combining **Retrieval-Augmented Generation (RAG)** for clinical Q&A with a **Deep Learning Ensemble** for skin disease classification — connected into a single diagnostic pipeline.

---

## 📌 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Part 1 — RAG Medical Q&A](#part-1--rag-based-medical-qa)
- [Part 2 — Skin Disease Classification](#part-2--skin-disease-classification)
- [Integration](#integration)
- [Features](#features)
- [Dataset](#dataset)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Future Work](#future-work)

---

## Overview

HEALBOT is a two-part intelligent medical system:

- **Part 1 (RAG)** — Fetches real-time clinical knowledge from PubMed, FDA, WHO, and ClinicalTrials.gov. Answers medical queries using a biomedical LLM with urgency scoring, location routing, multilingual support, and explainable AI.
- **Part 2 (Image)** — Classifies 23 skin diseases from dermoscopy images using an ensemble of EfficientNetV2-L and ConvNeXt-Base trained at 480×480 resolution.
- **Integration** — When Part 2 predicts a skin disease, the label is automatically passed to Part 1, which generates instant precautions, treatment guidance, and nearby facility directions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│            Text Query            Image Upload               │
└──────────────┬───────────────────────┬──────────────────────┘
               │                       │
               ▼                       ▼
┌──────────────────────┐   ┌───────────────────────────┐
│   PART 1 — RAG       │   │   PART 2 — IMAGE MODULE   │
│                      │   │                           │
│  Live APIs:          │   │  EfficientNetV2-L         │
│  • PubMed            │   │  + ConvNeXt-Base          │
│  • FDA               │   │  (480px Ensemble)         │
│  • WHO GHO           │   │         │                 │
│  • ClinicalTrials    │   │         ▼                 │
│                      │   │  Predicted Label          │
│  Local:              │   │  + Confidence Score       │
│  • MedQuAD.csv       │◄──┘                           │
│                      │   Label injected as context   │
│  S-PubMedBert        │                               │
│  Embeddings          │                               │
│       │              │                               │
│       ▼              │                               │
│  FAISS Vector DB     │                               │
│       │              │                               │
│       ▼              │                               │
│  GPT-mini-5 LLM      │                               │
└───────┬──────────────┘                               │
        │                                              │
        ▼
┌──────────────────────────────────────────────────────┐
│                   RESPONSE ENGINE                    │
│                                                      │
│  🟢 Home rest      📎 Cited Sources (XAI)            │
│  🟡 See a doctor   🌐 Hindi / English                │
│  🔴 Emergency      🛡️  Safety Guardrails             │
│                                                      │
│  📍 Google Maps API                                  │
│     🔴 → Nearest Hospital                           │
│     🟡 → Nearest Clinic                             │
│     🟢 → Nearest Pharmacy                           │
│     Auto-expands: 5 km → 10 km if no result         │
└──────────────────────────────────────────────────────┘
```

---

## Part 1 — RAG-Based Medical Q&A

The RAG pipeline fetches clinical data for **34 conditions** — 11 general diseases and all 23 DermNet skin disease categories — and stores them as searchable vectors in a FAISS database.

### Pipeline Files

| File | Role |
|---|---|
| `download_model.py` | Mounts Google Drive, loads cached BioMistral-7B or downloads it |
| `config.py` | Global config — model paths, DB paths, API credentials, embed model |
| `ingest.py` | Calls all APIs + loads MedQuAD.csv, outputs structured JSON |
| `build_db.py` | Chunks text → embeds via S-PubMedBert → stores in FAISS |
| `backend_rag.py` | MedicalBrain class — loads FAISS, queries GPT-mini-5, formats response |

### Embedding Model
[pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) — biomedical sentence encoder optimised for semantic similarity in clinical text.

### Why FAISS over ChromaDB
ChromaDB was computationally intensive for vector storage on limited hardware. FAISS provides faster approximate nearest-neighbour search with significantly lower memory overhead.

### Diseases Covered

**General (11):**
Eczema · Melanoma · Acne · Psoriasis · Diabetes · Hypertension · Asthma · Fever · Covid-19 · Dengue · Malaria

**DermNet Skin Categories (23):**
Actinic Keratosis · Basal Cell Carcinoma · Seborrheic Keratosis · Tinea Ringworm · and 19 others — activated automatically when Part 2 returns a classification.

---

## Part 2 — Skin Disease Classification

### Models

| Model | Parameters | Pretrained On | Native Resolution |
|---|---|---|---|
| EfficientNetV2-L | ~120 M | ImageNet-21k | 480 × 480 |
| ConvNeXt-Base | ~89 M | ImageNet-1k | 224 × 224 |

### Training Strategy

| Stage | Epochs | LR | Augmentation |
|---|---|---|---|
| Full Training | 30 | head=3e-4, backbone=3e-5 | RandAugment + MixUp + CutMix |
| Fine-Tuning | 10 | 3e-5 | OFF (clean signal) |
| SWA | 4 | 8e-6 | OFF |

**Additional techniques:**
- Resolution: 480 × 480 px
- Imbalance: WeightedRandomSampler + log-smoothed class weights
- Loss: CrossEntropy + label smoothing (0.10)
- Scheduler: CosineAnnealingLR (floor = lr × 0.05)
- Inference: 7-pass Test Time Augmentation (original, H-flip, V-flip, 90°, 180°, 270°, centre-crop)
- Final prediction: soft-vote ensemble (V2-L + ConvNeXt-Base)

### Training Curves

![Training Curves](outputs/training_curves.png)

EfficientNetV2-L reaches ~0.77 val accuracy by epoch 30. ConvNeXt-Base reaches ~0.71. Both models continue improving — fine-tuning and SWA extract additional gains before the final ensemble.

---

## Integration

When a user uploads a skin image:

```
Image → Ensemble Model → "Melanoma (87% confidence)"
                                ↓
                  Injected as context into MedicalBrain
                                ↓
              Instant response containing:
              • Precautions and treatment plan
              • Urgency tag (🟢🟡🔴)
              • Nearby clinic/hospital via Google Maps
              • Cited medical sources
              • Response in Hindi or English
```

No additional user input is required — the classification label triggers the full RAG response automatically.

---

## Features

| Feature | Description |
|---|---|
| 🟢🟡🔴 Urgency Scoring | Classifies severity — home rest / consult doctor / emergency |
| 📍 Smart Location | Google Maps routing to pharmacy, clinic, or hospital based on urgency. Auto-expands 5 km → 10 km |
| 🌐 Multilingual | Hindi and English input/output |
| 📎 Explainable AI | Every answer cites the exact source it was retrieved from |
| 🛡️ Safety Guardrails | Blocks harmful queries (self-harm, poison-related misuse) before reaching LLM |
| 🔗 Image-Text Pipeline | Skin disease label from Part 2 feeds directly into Part 1 RAG context |

---

## Dataset

| Property | Details |
|---|---|
| Name | DermNet |
| Source | [Kaggle — shubhamgoel27/dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) |
| Classes | 23 skin disease categories |
| Structure | ImageFolder (train / test split) |
| Challenge | Severe class imbalance across categories |

---

## Results

> Fill in after training completes

| Model | Accuracy | Balanced Acc | AUC-ROC (macro) |
|---|---|---|---|
| EfficientNetV2-L | — | — | — |
| EfficientNetV2-L + TTA | — | — | — |
| ConvNeXt-Base | — | — | — |
| ConvNeXt-Base + TTA | — | — | — |
| **Ensemble (V2-L + CNX + TTA)** | **—** | **—** | **—** |

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch 2.x · torchvision |
| LLM / RAG | LangChain · OpenAI GPT-mini-5 · BioMistral-7B |
| Embeddings | HuggingFace — S-PubMedBert-MS-MARCO |
| Vector DB | FAISS |
| Data APIs | PubMed · FDA · WHO GHO · ClinicalTrials.gov |
| Location | Google Maps API |
| Hardware | NVIDIA A100 80 GB |
| Language | Python 3.10 |

---

## Project Structure

```
HEALBOT/
│
├── part1_rag/
│   ├── download_model.py      # Cache BioMistral-7B to Google Drive
│   ├── config.py              # Global configuration
│   ├── ingest.py              # API calls + CSV loading
│   ├── build_db.py            # Chunk → embed → FAISS
│   └── backend_rag.py         # MedicalBrain class
│
├── part2_image/
│   ├── train.py               # Full training pipeline
│   ├── models.py              # EfficientNetV2-L + ConvNeXt-Base
│   └── inference.py           # TTA + ensemble prediction
│
├── outputs/
│   ├── best_efficientnetv2_l.pth
│   ├── best_convnext_base.pth
│   ├── class_map.json
│   ├── confusion_matrices.png
│   ├── per_class_f1.png
│   ├── model_comparison.png
│   └── training_curves.png
│
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/healbot.git
cd healbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up credentials
Edit `part1_rag/config.py`:
```python
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY      = "your_api_key"
OPENAI_API_KEY  = "your_openai_key"
GMAPS_API_KEY   = "your_google_maps_key"
EMAIL           = "your_email"   # required for PubMed API
```

### 4. Run Part 1 — Build RAG Database
```bash
python part1_rag/ingest.py        # Fetch data from APIs
python part1_rag/build_db.py      # Build FAISS vector database
```

### 5. Download DermNet and Train Part 2
```bash
# Dataset downloads automatically via Kaggle API
python part2_image/train.py
```

### 6. Run HEALBOT
```bash
python app.py
```

---

## Future Work

- [ ] Larger LLMs for higher answer quality
- [ ] Hybrid search (semantic + keyword) for better context recall
- [ ] Expand disease coverage beyond current 34 conditions
- [ ] Mobile app deployment
- [ ] Real-time video dermoscopy analysis

---

## License
This project is for academic and research purposes.

---

## Acknowledgements
- [DermNet Dataset — Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)
- [S-PubMedBert — HuggingFace](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO)
- [BioMistral-7B — HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B)
- PubMed · FDA · WHO · ClinicalTrials.gov for open medical APIs
