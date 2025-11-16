import sys
import os

# Ensure project root is on sys.path so `src` is importable when running via Streamlit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.resume_parser import (
    clean_text,
    extract_entities,
    compute_similarity,
    generate_feedback,
    extract_text,
)

st.set_page_config(page_title="Candidate Analysis Engine", layout="centered")
st.title("Candidate Analysis Engine")

uploaded_resumes = st.file_uploader(
    "Upload Resumes (.pdf / .docx / .txt)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

# text extraction logic moved to src.resume_parser.extractor.extract_text

job_description = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Enter or paste the job description here...",
)

if st.button("Compute Ranking"):
    if not uploaded_resumes:
        st.warning("Please upload at least one resume.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        job_clean = clean_text(job_description)
        job_entities = extract_entities(job_clean)

        results = []
        for file in uploaded_resumes:
            text = extract_text(file)
            resume_clean = clean_text(text)
            score = compute_similarity(resume_clean, job_clean)
            r_entities = extract_entities(resume_clean)
            feedback = generate_feedback({"entities": r_entities}, {"entities": job_entities})

            results.append({
                "Filename": file.name,
                "Score": round(score * 100, 2),
                "Feedback": feedback,
            })

        ranked = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
        st.subheader("Candidate Ranking")
        st.dataframe(ranked[["Filename", "Score"]])

        st.subheader("Top Match Visualization")
        top_k = st.slider("Select number of top resumes to visualize", 3, min(10, len(ranked)), 5)
        top = ranked.head(top_k)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(top["Filename"], top["Score"], color="steelblue")
        ax.set_xlabel("Match Score")
        ax.set_ylabel("Resume File")
        ax.invert_yaxis()
        ax.set_title("Top Resume Matches")
        st.pyplot(fig)

        best = top.iloc[0]
        st.subheader(f"Best Match: {best['Filename']}")
        st.success(f"Score: {best['Score']} / 100")
        st.info(best["Feedback"])
