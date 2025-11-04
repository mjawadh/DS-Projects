import re, spacy, torch
from sentence_transformers import SentenceTransformer, util

# load models once
try:
    nlp = spacy.load("saved_models/custom_ner_model")
except:
    nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
model.to(torch.float16)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_entities(text):
    doc = nlp(text)
    ents = {"NAME": None, "EMAIL": None, "PHONE": None, "SKILL": [], "EDUCATION": []}
    m = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if m: ents["EMAIL"] = m.group(0)
    p = re.search(r"\+?\d[\d\s-]{8,}\d", text)
    if p: ents["PHONE"] = p.group(0)
    for e in doc.ents:
        if e.label_ == "PERSON" and not ents["NAME"]:
            ents["NAME"] = e.text
        elif e.label_ in ["SKILL", "EDUCATION"]:
            ents[e.label_].append(e.text.lower())
    return ents

def compute_similarity(a, b):
    a_emb = model.encode(a, convert_to_tensor=True)
    b_emb = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(a_emb, b_emb))

def generate_feedback(resume, job):
    r_sk = set(resume["entities"]["SKILL"])
    j_sk = set(job["entities"]["SKILL"])
    missing = j_sk - r_sk
    msg = []
    if missing:
        msg.append(f"Add or highlight: {', '.join(missing)}.")
    if not resume["entities"]["EDUCATION"]:
        msg.append("Mention your degree or relevant coursework.")
    if not resume["entities"]["SKILL"]:
        msg.append("Include a clear skills section.")
    if not msg:
        msg.append("Good alignment — polish project descriptions.")
    return " ".join(msg)

print("helpers.py created successfully.")
