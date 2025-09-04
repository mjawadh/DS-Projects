import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import PyPDF2
import docx2txt
from rapidfuzz import fuzz   # NEW for fuzzy skill matching
from pdfminer.high_level import extract_text as pdfminer_extract
from docx import Document


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2
    ]
    return ' '.join(cleaned_tokens)


def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            page_text = re.sub(r"\s+", " ", page_text)
            text += page_text + '\n'
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
        text = re.sub(r"\s+", " ", text)  
    return text.strip()

# For better fit/matching increase Skills database
SKILLS_DB = {
    'Data Science': ['python', 'r', 'machine learning', 'data analysis', 'sql', 'pandas', 'numpy', 'scikit-learn', 'deep learning', 'tensorflow', 'pytorch', 'data modeling', 'data mining', 'tableau', 'power bi', 'data warehousing', 'survey data collection', 'predictive analytics', 'data visualization', 'sql server reporting services'],
    'HR': ['recruiting', 'human resources', 'onboarding', 'performance management', 'employee relations', 'talent acquisition'],
    'Web Designing': ['html', 'css', 'javascript', 'react', 'photoshop', 'ui', 'ux', 'figma', 'adobe xd', 'sketch'],
    'Software Engineer': ['java', 'c++', 'python', 'algorithms', 'data structures', 'aws', 'docker', 'git', 'kubernetes', 'sql'],
    'Business Analyst': ['business analysis', 'requirements gathering', 'stakeholder management', 'agile', 'sql', 'tableau', 'power bi', 'data visualization']
}

def normalize_text(text: str) -> str:
    """
    Normalize text for robust skill matching:
    - Lowercase
    - Replace -, _ with spaces
    - Collapse multiple spaces
    - Remove trailing numbers (e.g., python3 → python)
    """
    text = text.lower()
    text = re.sub(r"[-_]", " ", text)         # replace - and _ with space
    text = re.sub(r"\s+", " ", text)          # collapse multiple spaces
    text = re.sub(r"(\w+)\d+", r"\1", text)   # remove trailing numbers from words
    return text.strip()

def clean_resume_lines(resume_text: str):
    lines = []
    for line in resume_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove bullets or dashes at start (added '·' and '●')
        line = re.sub(r"^[•\-–\*·●]\s*", "", line)
        lines.append(normalize_text(line))
    return lines

def calculate_fit_score(resume_text, job_category, threshold=85):
    if job_category not in SKILLS_DB:
        return 0, []

    required_skills = SKILLS_DB[job_category]

    search_text_normalized = normalize_text(resume_text)
    words_in_resume = search_text_normalized.split()
    lines_in_resume = clean_resume_lines(resume_text)

    matched_skills = []
    for skill in required_skills:
        skill_normalized = normalize_text(skill)

        # 1. Short skills (<=3 chars): exact word match only
        if len(skill_normalized) <= 3:
            if skill_normalized in words_in_resume:
                matched_skills.append(skill)
            continue

        # 2. Exact substring anywhere in resume
        if skill_normalized in search_text_normalized:
            matched_skills.append(skill)
            continue

        # 3. Line-level fuzzy match (for list-style resumes)
        for line in lines_in_resume:
            if fuzz.partial_ratio(skill_normalized, line) >= threshold:
                matched_skills.append(skill)
                break

    matched_skills = list(set(matched_skills))
    score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
    return int(score), matched_skills


import joblib
import streamlit as st
import os

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'best_resume_classifier.joblib')
vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.joblib')
label_encoder_path = os.path.join(script_dir, 'label_encoder.joblib')

try:
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    st.write("Models loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}. Ensure all .joblib files are in {script_dir}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit App UI
st.set_page_config(layout="wide")
st.title("🤖 Automated Resume Screening App")
st.markdown("Upload a resume in PDF or DOCX format to see the predicted job role and its relevance.")

uploaded_file = st.file_uploader("Attach a resume to analyze", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner('Analyzing the resume...'):
        # 1. Extract text from the uploaded file
        resume_text = extract_text(uploaded_file)
        
        # 2. Preprocess the extracted text (for the model prediction only)
        cleaned_resume_for_model = preprocess_text(resume_text)
        
        # 3. Vectorize the text
        resume_tfidf = tfidf_vectorizer.transform([cleaned_resume_for_model])
        
        # 4. Predict the job category
        prediction_id = model.predict(resume_tfidf)[0]
        predicted_category = label_encoder.inverse_transform([prediction_id])[0]
        
        # 5. Calculate Fit Score and find matched skills
        fit_score, matched_skills = calculate_fit_score(resume_text, predicted_category)

    st.success("Analysis Complete!")

    st.header("Analysis Results")
    st.markdown(f"**Predicted Job Role:** `{predicted_category}`")
    st.markdown(f"**Fit Score:** `{fit_score}%`")
        
    st.subheader("Matched Skills")
    if matched_skills:
        for skill in matched_skills:
            st.write(f"- {skill.capitalize()}")
    else:
        st.write("No specific skills from our database were matched for this role.")

    st.header("Extracted Resume Text")
    st.code(resume_text, language="text")
