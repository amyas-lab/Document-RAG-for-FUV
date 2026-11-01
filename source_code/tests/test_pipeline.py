"""Offline tests for PDF loading and chunking (no Ollama / Qdrant needed)."""

import shutil

import pytest

from rag.chunking import chunk_documents, course_codes
from rag.config import load_config
from rag.loaders import clean_text, load_documents, toc_titles

HAS_TESSERACT = shutil.which("tesseract") is not None


def test_clean_text_removes_toc_and_markup():
    raw = "|MATH 301: Optimization ........................ 24|\n|---|\n**Bold** <mark>x</mark>\n12\nBody"
    assert clean_text(raw) == "Bold x\nBody"


def test_toc_titles():
    assert toc_titles(["Minor Requirements (24 credits) ........ 9", "plain line"]) == ["Minor Requirements (24 credits)"]


def test_course_codes():
    assert course_codes("MATH\n101 and MATH 305; ECON 209") == ["ECON 209", "MATH 101", "MATH 305"]


@pytest.fixture(scope="module", params=["pypdf", "pymupdf"])
def pages(request):
    cfg = load_config().with_overrides(pdf_parser=request.param, ocr_enabled=HAS_TESSERACT)
    return cfg, load_documents(cfg)


def test_pdf_pages_have_metadata(pages):
    cfg, docs = pages
    assert len(docs) >= 25
    assert all(d.metadata["source"] == "MATH_MajorHandbook.pdf" and d.metadata["page"] >= 1 for d in docs)
    assert not any("......" in d.page_content for d in docs), "TOC dot leaders must be stripped"


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract not installed")
def test_flowchart_page_is_ocred(pages):
    _, docs = pages
    page14 = [d for d in docs if d.metadata["page"] == 14]
    assert page14 and page14[0].metadata["ocr"]
    assert "FLOWCHART" in page14[0].page_content.upper()


def test_structure_chunking_one_chunk_per_course():
    cfg = load_config().with_overrides(pdf_parser="pymupdf", chunking="structure")
    chunks = chunk_documents(load_documents(cfg), cfg)
    sections = {c.metadata["section"] for c in chunks}
    assert any(s.endswith("MATH 305: Stochastic Calculus") for s in sections)
    assert any(s.endswith("Minor Requirements (24 credits)") for s in sections)
    assert any("5.2. Student Assessment" in s and "5.1." not in s for s in sections)
    assert max(len(c.page_content) for c in chunks) <= cfg.chunk_size
