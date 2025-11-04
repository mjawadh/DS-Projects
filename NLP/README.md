# Resume Parser & JobMatch Scorer (NLP)

## Project Overview
This repository contains code and resources for parsing resumes and matching them to job descriptions using NLP and NER techniques. The workflow is implemented in Python and leverages libraries such as HuggingFace Transformers, pandas, and scikit-learn.

## Folder Structure
```
NLP/
├── .gitignore
├── helpers.py           # Utility functions for data processing
├── app.py               # Application
├── proj9.ipynb          # Main notebook for EDA, NER, training logic and experiments
├── requirements.txt     # Python dependencies
├── temp/                # Python environment and cache files
```

## How to Use
1. **Create Environment**
   - Run `python -m venv venv` to create virtual environment.
   - Run `venv\Scripts\activate` to activate the environment.
1. **Install Dependencies**
   - Run `pip install -r requirements.txt` to install required packages.
2. **Explore the Notebook**
   - Open `proj9.ipynb` for exploratory data analysis, NER experiments, and workflow steps, model training and inference scripts.
3. **NER Model & Utilities**
   - Use `app.py` for launching the application.
   - Use `helpers.py` for data loading and preprocessing utilities.

## Key Files
- `proj9.ipynb`: Main workflow and experiments.
- `app.py`: NER model code and training functions.
- `helpers.py`: Data processing utilities.
- `requirements.txt`: List of required Python packages.

## Notes
- All data files and outputs should be placed in appropriate subfolders as referenced in the notebook or scripts.
- The workflow and scripts are designed for educational and research purposes.

## License
This project is for educational and research use. See individual files for license details.
