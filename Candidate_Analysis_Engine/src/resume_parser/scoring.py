import json
import re
import math
import hashlib
import importlib

# Lazy model holder. We avoid importing heavy libraries at module import time so tests
# can run in environments without the sentence-transformers package installed.
_model = None
_util = None


def _ensure_model():
    """Ensure the sentence-transformers model is loaded. If the package is not
    available, create a lightweight fallback that produces deterministic embeddings
    for smoke testing.
    """
    global _model, _util
    if _model is not None and _util is not None:
        return _model, _util

    try:
        st = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(st, "SentenceTransformer")
        util = importlib.import_module("sentence_transformers.util")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _util = util
        return _model, _util
    except Exception:
        # Fallback: deterministic hash-based embeddings
        class _FallbackModel:
            def encode(self, texts, convert_to_tensor=False):
                if isinstance(texts, str):
                    texts = [texts]
                embs = []
                for t in texts:
                    # create a small fixed-size vector from SHA1
                    h = hashlib.sha1(t.encode('utf-8')).digest()
                    # convert bytes to floats in range [-1,1]
                    vec = [((b / 255.0) * 2.0 - 1.0) for b in h[:16]]
                    embs.append(vec)
                return embs[0] if len(embs) == 1 else embs

        class _FallbackUtil:
            @staticmethod
            def cos_sim(a, b):
                # Accept lists or list-like vectors
                def to_vec(x):
                    if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                        return list(x)
                    return [float(x)]

                va = to_vec(a)
                vb = to_vec(b)
                # pad to same length
                n = max(len(va), len(vb))
                va = va + [0.0] * (n - len(va))
                vb = vb + [0.0] * (n - len(vb))
                dot = sum(x * y for x, y in zip(va, vb))
                na = math.sqrt(sum(x * x for x in va))
                nb = math.sqrt(sum(x * x for x in vb))
                if na == 0 or nb == 0:
                    return 0.0
                return dot / (na * nb)

        _model = _FallbackModel()
        _util = _FallbackUtil()
        return _model, _util


def compute_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two texts.

    Uses sentence-transformers if available, otherwise uses a deterministic
    fallback embedding for smoke tests.
    """
    model, util = _ensure_model()
    a_emb = model.encode(a, convert_to_tensor=False)
    b_emb = model.encode(b, convert_to_tensor=False)
    sim = util.cos_sim(a_emb, b_emb)
    # util.cos_sim may return a tensor-like value or a float
    try:
        return float(sim)
    except Exception:
        return sim


def generate_feedback(resume: dict, job: dict) -> str:
    """Generate simple, actionable feedback based on missing skills/education.

    Args:
        resume: dict with key 'entities' mapping to entity lists (e.g., {'SKILL': [...], 'EDUCATION': [...]})
        job: dict with same structure

    Returns:
        A short string with suggestions.
    """
    r_sk = set(resume.get("entities", {}).get("SKILL", []))
    j_sk = set(job.get("entities", {}).get("SKILL", []))
    missing = j_sk - r_sk
    msg = []
    if missing:
        msg.append(f"Add or highlight: {', '.join(sorted(missing))}.")
    if not resume.get("entities", {}).get("EDUCATION"):
        msg.append("Mention your degree or relevant coursework.")
    if not resume.get("entities", {}).get("SKILL"):
        msg.append("Include a clear skills section.")
    if not msg:
        msg.append("Good alignment — polish project descriptions.")
    return " ".join(msg)


if __name__ == "__main__":
    # Simple smoke CLI for manual testing
    import argparse

    parser = argparse.ArgumentParser(description="Run simple scoring on provided texts")
    parser.add_argument("--resume", type=str, help="Resume text or path to text file")
    parser.add_argument("--job", type=str, help="Job description text or path to text file")
    args = parser.parse_args()

    def _load_text(s):
        try:
            with open(s, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return s

    resume_text = _load_text(args.resume)
    job_text = _load_text(args.job)
    print('Similarity:', compute_similarity(resume_text, job_text))
    print('Feedback:', generate_feedback({'entities': {}}, {'entities': {}}))
