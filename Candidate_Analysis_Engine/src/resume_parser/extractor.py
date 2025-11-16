import pdfplumber
import docx2txt


def extract_text(file):
    """Extract text from uploaded resume file-like object.
    Supports PDF, DOCX/DOC, and plain text.
    """
    name = getattr(file, 'name', '').lower()
    if name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return '\n'.join(page.extract_text() or '' for page in pdf.pages)
    elif name.endswith(('.docx', '.doc')):
        return docx2txt.process(file)
    else:
        # file may be an uploaded stream: read and decode
        try:
            return file.read().decode('utf-8', errors='ignore')
        except Exception:
            return str(file)
