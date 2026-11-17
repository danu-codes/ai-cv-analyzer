import fitz  # PyMuPDF
import docx
import re

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text block-by-block to preserve reading order in multi-column CVs."""
    text_blocks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            # Sort blocks top-to-bottom, left-to-right
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
            for block in blocks:
                if block[6] == 0:  # 0 indicates a text block (ignores images)
                    text_blocks.append(block[4].strip())
    return clean_text("\n".join(text_blocks))

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts paragraphs from a .docx file."""
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(paragraphs))

def clean_text(raw_text: str) -> str:
    """Normalizes whitespace and removes non-printable artifacts."""
    text = re.sub(r'\r\n|\r', '\n', raw_text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()