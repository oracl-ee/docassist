from pathlib import Path

from pypdf import PdfReader

SUPPORTED = (".txt", ".md", ".pdf")


def load_file(path):
    """Read one .txt, .md, or .pdf file. Returns {"source", "text"} or None."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return None
    if not text.strip():
        return None
    return {"source": path.name, "text": text}


def load_documents(folder="data"):
    docs = []
    for path in sorted(Path(folder).iterdir()):
        doc = load_file(path)
        if doc:
            docs.append(doc)
    return docs