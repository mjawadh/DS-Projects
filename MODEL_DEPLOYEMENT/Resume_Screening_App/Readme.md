# Resume Screening App

## Overview
This repository contains a **Resume Screening App** designed to automate the process of screening resumes for various job roles. The application leverages Natural Language Processing (NLP) and machine learning to classify resumes into job categories (e.g., Data Science, Software Engineer, Web Developer, HR, Testing) and predict a fit score based on the candidate's skills and experience. It provides a user-friendly interface using Streamlit, allowing recruiters to upload resumes (PDF or DOCX) and view results instantly.

## Features
- **Resume Ingestion**: Upload resumes in PDF or DOCX format.
- **Text Preprocessing**: Cleans and preprocesses resume text using NLP techniques (e.g., tokenization, lemmatization, stopword removal).
- **Job Role Classification**: Predicts the most suitable job role using a trained Logistic Regression model.
- **Fit Score Prediction**: Calculates a percentage match score for the predicted role.
- **Skill Matching**: Identifies key skills aligned with the job role.
- **Deployment Ready**: Designed for deployment on Streamlit Cloud or similar platforms.

## Project Structure
- `app.py` -- Main Streamlit application script.
- `best_resume_classifier.joblib` -- Trained Logistic Regression model file.
- `tfidf_vectorizer.joblib` -- Trained TF-IDF vectorizer file.
- `label_encoder.joblib` -- Label encoder for job categories (if used).
- `preprocessed_resumes.csv` -- Preprocessed resume dataset (optional for reference).
- `proj7.ipynb` -- Jupyter notebook with project development and testing code.
- `requirements.txt` -- List of Python dependencies.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mjawadh/DS-Projects/tree/main/MODEL_DEPLOYEMENT/Resume_Screening_App.git
   cd Resume_Screening_App
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure all `.joblib` model files and `preprocessed_resumes.csv` are in the root directory.

## Usage
 1. Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```
 2. Upload a resume file (PDF or DOCX) using the file uploader.
 3. View the predicted job role, fit score, and matched skills.

## Deployment
**Streamlit Cloud**: 
    1. Push the repository to GitHub.
    2. Connect the repo to Streamlit Community Cloud.
    3. Set `app.py` as the main file and deploy.
 Ensure all `.joblib` files are committed to the repo for the app to load models correctly.

## Dataset
The app uses an open-source Resume Dataset from Kaggle:
- **Link**: [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- **Details**: Contains ~1000+ resumes labeled with job categories (e.g., Data Science, HR, Software Engineering).

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Web app framework.
- **Scikit-learn**: Machine learning and TF-IDF vectorization.
- **NLTK**: NLP preprocessing.
- **PyPDF2**: PDF text extraction.
- **python-docx**: DOCX text extraction.
- **Joblib**: Model serialization.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the app. Contributions are welcome!

## Acknowledgments
- Inspired by the need to automate resume screening for HR teams.
- Thanks to the Kaggle community for the dataset.