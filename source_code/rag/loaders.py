"""Document loading (Task 1): .txt and .pdf -> one LangChain Document per page.

Two PDF parsers are supported so they can be compared in the report:
  * ``pypdf``   - baseline suggested by the assignment; plain text, tables lose structure.
  * ``pymupdf`` - pymupdf4llm Markdown output; keeps headings and renders tables as
                  Markdown tables, which enables heading-aware chunking.

Pages without a text layer (scanned pages, diagrams such as the handbook's flowchart on
page 14) are rendered to an image and passed through Tesseract OCR when enabled.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

from .config import Config

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt"}
OCR_NOTE = "(Text extracted by OCR from an image-only page; it may contain recognition errors.)"


# --------------------------------------------------------------------------- utils
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_documents(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file())


# --------------------------------------------------------------------------- cleaning
_DOT_LEADER = re.compile(r"\.{5,}\s*\d+\s*\|?\s*$")     # TOC lines: "MATH 301: Optimization ...... 24" (or "...24|")
_TABLE_RULE = re.compile(r"^\|[-|: ]+\|$")              # Markdown table separator "|---|---|"
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")           # stray page-number lines
_HTML_TAG = re.compile(r"</?(?:mark|u|sup|sub|span|i|b|em|strong)[^>]*>", re.I)
_PICTURE_BLOCK = re.compile(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", re.S)


def clean_text(text: str) -> str:
    """Remove layout noise that hurts retrieval while keeping Markdown structure."""
    text = _PICTURE_BLOCK.sub("", text)
    text = text.replace("<br>", " ")
    text = _HTML_TAG.sub("", text)
    text = text.replace("**", "")                        # bold markers add tokens, no meaning
    text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    lines: list[str] = []
    for line in text.splitlines():
        if _DOT_LEADER.search(line) or _PAGE_NUMBER.match(line):
            continue
        # drop a table separator left orphaned after its header row was removed (TOC tables)
        if _TABLE_RULE.match(line.strip()) and not (lines and lines[-1].lstrip().startswith("|")):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def toc_titles(raw_pages: list[str]) -> list[str]:
    """Heading titles listed in the document's table of contents (lines with dot leaders).

    Word exports often render some headings as plain text (no font-size change), so the
    chunker uses these titles to recognise headings that the parser did not mark."""
    titles = []
    for page in raw_pages:
        for line in page.splitlines():
            if _DOT_LEADER.search(line):
                title = re.sub(r"\.{5,}.*$", "", line)
                title = re.sub(r"[|*#_]", "", title).strip()
                if len(title) > 3:
                    titles.append(title)
    return titles


# --------------------------------------------------------------------------- OCR
def ocr_page(pdf_path: Path, page_index: int, dpi: int = 300) -> str:
    """Render one PDF page with PyMuPDF and OCR it with Tesseract."""
    import io

    import pymupdf
    import pytesseract
    from PIL import Image

    with pymupdf.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, config="--psm 3")


def _raw_text_lengths(pdf_path: Path) -> list[int]:
    """Characters in each page's text layer (used to detect image-only pages)."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [len(page.get_text().strip()) for page in doc]


# --------------------------------------------------------------------------- parsers
def _pages_pypdf(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]


def _pages_pymupdf(pdf_path: Path) -> list[str]:
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(
        str(pdf_path), page_chunks=True, use_ocr=False, header=False, footer=False, show_progress=False
    )
    return [p["text"] for p in pages]


PARSERS = {"pypdf": _pages_pypdf, "pymupdf": _pages_pymupdf}


def load_pdf(pdf_path: Path, cfg: Config) -> list[Document]:
    if cfg.pdf_parser not in PARSERS:
        raise ValueError(f"Unknown pdf_parser '{cfg.pdf_parser}'. Choose from {list(PARSERS)}")
    pages = PARSERS[cfg.pdf_parser](pdf_path)
    raw_lengths = _raw_text_lengths(pdf_path)
    toc = tuple(toc_titles(pages))
    docs: list[Document] = []

    for i, text in enumerate(pages):
        page_no = i + 1
        ocr_used = False
        if raw_lengths[i] < cfg.ocr_min_chars:
            if cfg.ocr_enabled:
                try:
                    text = OCR_NOTE + "\n" + ocr_page(pdf_path, i, cfg.ocr_dpi)
                    ocr_used = True
                    log.info("  page %d: no text layer -> OCR (%d chars)", page_no, len(text))
                except Exception as exc:  # tesseract missing, etc.
                    log.warning("  page %d: OCR failed (%s); page skipped", page_no, exc)
                    continue
            else:
                log.warning("  page %d: no text layer and OCR disabled; page skipped", page_no)
                continue

        text = clean_text(text)
        if len(text) < 20:  # e.g. a TOC page that became empty after cleaning
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_no,
                    "parser": cfg.pdf_parser,
                    "ocr": ocr_used,
                    "format": "markdown" if cfg.pdf_parser == "pymupdf" and not ocr_used else "text",
                    "toc_titles": toc,  # consumed by the chunker, not stored in the index
                },
            )
        )
    return docs


def load_txt(txt_path: Path) -> list[Document]:
    text = clean_text(txt_path.read_text(encoding="utf-8"))
    return [Document(page_content=text, metadata={"source": txt_path.name, "page": 1, "parser": "txt",
                                                  "ocr": False, "format": "text"})]


def load_documents(cfg: Config) -> list[Document]:
    """Load every supported file in ``cfg.data_dir`` into per-page Documents."""
    files = find_documents(cfg.data_path)
    if not files:
        raise FileNotFoundError(f"No .pdf or .txt files found in {cfg.data_path}")
    docs: list[Document] = []
    for path in files:
        log.info("Loading %s", path.name)
        loaded = load_pdf(path, cfg) if path.suffix.lower() == ".pdf" else load_txt(path)
        log.info("  -> %d pages with text", len(loaded))
        docs.extend(loaded)
    return docs
