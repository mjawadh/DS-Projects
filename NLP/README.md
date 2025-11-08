# Resume Parser & JobMatch Scorer (NLP)

---

## Project Overview
This repository contains code and resources for parsing resumes and matching them to job descriptions using NLP and NER techniques. The workflow is implemented in Python and leverages libraries such as HuggingFace Transformers, pandas, and scikit-learn.

---

## Table of contents
- About
- Quick demo (notebooks)
- Key features
- Limitations
- Tech stack & model details
- System requirements
- Architecture & workflow
- Repository layout
- Installation
- Usage
- Data & caching notes
- Evaluation
- Future work
- Contributing

---

## About
This project extracts structured information from raw resumes (skills, education, experience, contact info, etc.) using Named Entity Recognition (NER) and applies simple matching/scoring logic to rank resumes against job descriptions. The code is intended for experiments, research, and teaching — not a production-ready service.

---

## Quick demo — notebooks
Open the primary notebook `proj9.ipynb` to see an end-to-end interactive exploration: data loading, EDA, dataset QC, label mapping, token/label alignment utilities and example training snippets.

---

## Key features
- Resume parsing with token-level entity extraction (NER).
- Quick dataset QC utilities (column inspection, label mapping, sample display).
- Helper scripts for preprocessing and model training (`helpers.py`, `ner.py`).
- Lightweight smoke-training harness for pipeline validation inside the notebook.

---

## Limitations
- Not production hardened: no authentication, rate-limiting, or REST API.
- Model training is experimental and may require GPU for reasonable speed.
- No formal CI tests or scoring suite included by default — add tests before production use.

---

## Tech stack & model details
- Language: Python (3.8+ recommended)
- Models & libs: `transformers` (HuggingFace), `datasets`, `torch`/`accelerate` (optional), `scikit-learn`, `pandas`, `seqeval`
- Tokenizer: BERT-family (notebook uses `bert-base-uncased` by default for inspection)
- Typical model: `bert-base-uncased` (fine-tune for token classification / NER)

Refer to `requirements.txt` for a pinned list of packages used during development.

---

## System requirements
- Python 3.8+ (3.10 recommended)
- 8+ GB RAM for local experiments; GPU recommended for training (CUDA-enabled PyTorch)

---

## Architecture & workflow
High-level flow:
1. Raw resume CSVs are loaded and inspected in `proj9.ipynb`.
2. Data quality checks and label discovery identify which columns represent entities.
3. Labels are converted to a consistent mapping (`ner_label_maps.json` produced by helper cells).
4. Text is tokenized with a HuggingFace tokenizer and aligned to IOB token-level labels.
5. Use the training helper (`fine_tune_ner_model_robust`) for smoke-training and validation.
6. Parsed entities are used as features for lightweight matching/scoring against job descriptions (notebook contains examples).

---

## Repository layout
```
NLP/
├── proj9.ipynb                # Main notebook: EDA, dataset QC, NER pipeline experiments
├── ner.py                     # NER training/inference helper functions
├── helpers.py                 # Data loading, preprocessing, token/label alignment utilities
├── requirements.txt           # Python dependencies used in development
├── README.md                  # This file
└── temp/                      # Local virtualenv and cached artifacts (not checked in)
```

---

## Installation
Recommended to use a virtual environment.

Windows (cmd.exe):
```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

If you plan to run training on GPU, install a matching `torch` build with CUDA support.

---

## Usage
Explore and run the notebook `proj9.ipynb` for step-by-step usage. Typical tasks:

- Open the notebook and run the EDA / dataset QC cells to discover label columns and inspect samples.
- Run the label map creation cell to produce `ner_label_maps.json` (used by training routines).
- Prepare token-level IOB labels and tokenization using the helper utilities.
- Run a smoke training (1 epoch) using `fine_tune_ner_model_robust` to validate the pipeline.

Notes:
- The notebook includes defensive code paths for loading CSVs and converting to HuggingFace `Dataset` objects.
- Replace dataset paths in the notebook cells to point to your local data/cache if needed.

---

## Data & caching notes
- Raw datasets should be stored outside the repository and referenced by absolute or relative paths in the notebook.
- If you use external downloads (e.g., Kaggle), ensure credentials and caches are set up for `kagglehub` or `kaggle` CLI.

---

## Evaluation
- The notebook includes basic evaluation helpers (e.g., `seqeval` for token-level F1 and classification reports).
- For production evaluation, add a held-out validation split, compute per-entity precision/recall/F1 and confusion analysis.

---

## Future work
- Add a small CLI or Streamlit UI for quick resume uploads and parsing.
- Create a reproducible training script and a lightweight test suite.
- Add model serialization for inference and a simple matching API.

---

## Contributing
Contributions are welcome. Suggested workflow:
1. Fork the repository and create a feature branch.
2. Run the notebook and any updated scripts locally.
3. Open a PR with a clear description and link to any relevant issue.
