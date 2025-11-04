# app.py
import streamlit as st
import pdfplumber, docx2txt
import pandas as pd
import matplotlib.pyplot as plt
from helpers import clean_text, extract_entities, compute_similarity, generate_feedback

st.set_page_config(page_title="Resume Ranking & Match Scorer", layout="centered")
st.title("Resume ↔ Job Match Scorer & Ranking System")

# ---------- 1. Upload multiple resumes ----------
uploaded_resumes = st.file_uploader(
    "Upload Resumes (.pdf / .docx / .txt)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

def extract_text(file):
    """Extract text content from uploaded resume file."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif name.endswith((".docx", ".doc")):
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8", errors="ignore")

# ---------- 2. Paste Job Description ----------
job_description = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Enter or paste the job description here..."
)

# ---------- 3. Compute Matches ----------
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
                "Feedback": feedback
            })

        # ---------- 4. Ranking ----------
        ranked = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
        st.subheader("Candidate Ranking")
        st.dataframe(ranked[["Filename", "Score"]])

        # ---------- 5. Bonus Visualization ----------
        st.subheader("Top Match Visualization")
        top_k = st.slider("Select number of top resumes to visualize", 3, min(10, len(ranked)), 5)
        top = ranked.head(top_k)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(top["Filename"], top["Score"], color="skyblue")
        ax.set_xlabel("Match Score")
        ax.set_ylabel("Resume File")
        ax.invert_yaxis()
        ax.set_title("Top Resume Matches")
        st.pyplot(fig)

        # optional: detailed feedback for top resume
        best = top.iloc[0]
        st.subheader(f"Best Match: {best['Filename']}")
        st.success(f"Score: {best['Score']} / 100")
        st.info(best["Feedback"])
