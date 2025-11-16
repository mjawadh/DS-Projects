import re
import importlib
from typing import Dict, Any

# Lazy-load heavy NLP libraries (spaCy, sentence-transformers). If unavailable,
# fall back to lightweight, deterministic heuristics so the system can be smoke-tested
# without large downloads.

_nlp = None


def _ensure_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        spacy = importlib.import_module('spacy')
        try:
            _nlp = spacy.load('saved_models/custom_ner_model')
        except Exception:
            try:
                _nlp = spacy.load('en_core_web_sm')
            except Exception:
                _nlp = None
    except Exception:
        _nlp = None
    return _nlp


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract basic entities from text.

    Uses spaCy NER if available; otherwise uses regex and simple heuristics.
    Returns a dict with keys: NAME, EMAIL, PHONE, SKILL (list), EDUCATION (list)
    """
    nlp = _ensure_spacy()
    ents = {"NAME": None, "EMAIL": None, "PHONE": None, "SKILL": [], "EDUCATION": []}

    # Email
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if m:
        ents['EMAIL'] = m.group(0)

    # Phone number (simple extraction)
    p = re.search(r'\+?\d[\d\s\-]{7,}\d', text)
    if p:
        ents['PHONE'] = p.group(0)

    # If spaCy NER available, use it for PERSON/SKILL/EDUCATION where possible
    if nlp is not None:
        try:
            doc = nlp(text)
            for e in doc.ents:
                lbl = getattr(e, 'label_', '').upper()
                if lbl == 'PERSON' and not ents['NAME']:
                    ents['NAME'] = e.text
                elif lbl in ('SKILL', 'EDUCATION'):
                    ents.setdefault(lbl, []).append(e.text.lower())
        except Exception:
            # fallback to heuristics below
            pass

    # Heuristic skill extraction: look for common skill keywords if none found
    if not ents['SKILL']:
        skill_keywords = ['python', 'pandas', 'machine learning', 'sql', 'tensorflow', 'pytorch', 'react', 'javascript']
        lowered = text.lower()
        for kw in skill_keywords:
            if kw in lowered:
                ents['SKILL'].append(kw)

    # Heuristic education extraction
    if not ents['EDUCATION']:
        edu_keywords = ['bachelor', "b\.sc", 'master', 'msc', 'phd', 'degree']
        for kw in edu_keywords:
            if kw in text.lower():
                ents['EDUCATION'].append(kw)

    return ents


print('preprocessing module loaded (lazy NLP imports).')
