"""Chunking strategies.

* ``fixed``     - baseline: RecursiveCharacterTextSplitter (chunk_size / chunk_overlap), applied
                  per page so every chunk keeps an exact page number.
* ``structure`` - heading-aware: the Markdown produced by pymupdf4llm is split into sections
                  (e.g. "4. Program Structure and Curriculum > Minor Requirements (24 credits)"),
                  each course description becomes its own chunk, tables are never cut in the
                  middle of a row (long tables are split by rows with the header repeated), and
                  the section path is prepended to the chunk text so that the embedding "knows"
                  where the text comes from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config

COURSE_CODE = re.compile(r"\b(MATH|CS|ECON|IS)\s*(\d{3})\b")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+[A-Z][^.]{2,70}$")   # "5.1. Teaching and Learning Approach"
_NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _level_from_number(title: str, default: int) -> int:
    """'5. X' -> 2, '5.1. X' -> 3, '5.2.5. X' -> 4 (consistent regardless of font size)."""
    m = _NUMBER_PREFIX.match(title)
    return 2 + m.group(1).count(".") if m else default
_NOT_A_HEADING = re.compile(r"^(pre-?requisites?|cross-?list|note)\b", re.I)


def course_codes(text: str) -> list[str]:
    return sorted({f"{a} {b}" for a, b in COURSE_CODE.findall(text)})


# --------------------------------------------------------------------------- fixed
def fixed_chunks(pages: list[Document], cfg: Config) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata.pop("toc_titles", None)
        c.metadata.update(section="", course_codes=course_codes(c.page_content), chunk_index=i, chunking="fixed")
    return chunks


# --------------------------------------------------------------------------- structure
@dataclass
class _Block:
    text: str
    page: int
    ocr: bool
    is_table: bool


@dataclass
class _Section:
    source: str
    path: tuple[str, ...]
    blocks: list[_Block] = field(default_factory=list)


def _detect_heading(line: str, toc: set[str]) -> tuple[int, str] | None:
    """Return (level, title) if the line is a section heading.

    Three signals: Markdown '#' headings from the parser, numbered headings ("5.1. ..."),
    and plain lines that exactly match a table-of-contents entry."""
    s = line.strip()
    m = _MD_HEADING.match(s)
    if m:
        title = m.group(2).strip(" _*:")
        if not title or _NOT_A_HEADING.match(title):
            return None
        return _level_from_number(title, len(m.group(1))), title
    m = _NUMBERED_HEADING.match(s)
    if m:
        return 2 + m.group(1).count("."), s.rstrip(":")
    if 3 < len(s) < 100 and _norm(s) in toc:
        return _level_from_number(s, 3), s
    return None


def _build_sections(pages: list[Document]) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current: _Section | None = None
    last_source = None

    def start(source: str):
        nonlocal current
        current = _Section(source=source, path=tuple(t for _, t in stack if t))
        sections.append(current)

    for doc in pages:
        source, page, ocr = doc.metadata["source"], doc.metadata["page"], doc.metadata.get("ocr", False)
        toc = {_norm(t) for t in doc.metadata.get("toc_titles", ())}
        if source != last_source:  # new file -> reset heading stack
            stack, last_source = [], source
            start(source)

        lines = doc.page_content.splitlines()
        if ocr:
            # An image-only page gets its own section titled by its first meaningful OCR line.
            title = next((l.strip() for l in lines[1:] if len(l.strip()) > 8), f"Image page {page}")
            stack = [s for s in stack if s[0] <= 2] + [(3, f"{title} (page {page}, OCR)")]
            start(source)

        para: list[str] = []
        para_is_table = False

        def flush():
            nonlocal para, para_is_table
            text = "\n".join(para).strip()
            if text:
                current.blocks.append(_Block(text, page, ocr, para_is_table))
            para, para_is_table = [], False

        for line in lines:
            heading = None if ocr else _detect_heading(line, toc)
            if heading:
                flush()
                level, title = heading
                if level == 1:          # document title ("Undergraduate Major Handbook")
                    continue
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                start(source)
                continue
            if not line.strip():
                flush()
                continue
            is_table_line = line.lstrip().startswith("|")
            if para and is_table_line != para_is_table:
                flush()
            para_is_table = is_table_line
            # demote non-heading "#### Pre-requisites: ..." lines to plain text
            para.append(re.sub(r"^#{1,6}\s+", "", line))
        flush()  # never let a block span two pages -> exact page citations
    return [s for s in sections if s.blocks]


def _split_table(block: _Block, limit: int) -> list[_Block]:
    rows = block.text.splitlines()
    header, body = rows[:2], rows[2:]
    out, cur = [], list(header)
    for row in body:
        if len("\n".join(cur + [row])) > limit and len(cur) > len(header):
            out.append(_Block("\n".join(cur), block.page, block.ocr, True))
            cur = list(header)
        cur.append(row)
    out.append(_Block("\n".join(cur), block.page, block.ocr, True))
    return out


def _split_long(block: _Block, cfg: Config, limit: int) -> list[_Block]:
    if len(block.text) <= limit:
        return [block]
    if block.is_table:
        return _split_table(block, limit)
    splitter = RecursiveCharacterTextSplitter(chunk_size=limit, chunk_overlap=cfg.chunk_overlap)
    return [_Block(t, block.page, block.ocr, False) for t in splitter.split_text(block.text)]


def structure_chunks(pages: list[Document], cfg: Config) -> list[Document]:
    chunks: list[Document] = []
    base_meta = {k: pages[0].metadata[k] for k in ("parser",)} if pages else {}
    for section in _build_sections(pages):
        header = f"Section: {' > '.join(section.path)}" if section.path else ""
        limit = max(200, cfg.chunk_size - len(header) - 2)
        blocks = [b for blk in section.blocks for b in _split_long(blk, cfg, limit)]

        group: list[_Block] = []

        def emit():
            if not group:
                return
            body = "\n\n".join(b.text for b in group)
            text = f"{header}\n\n{body}" if header else body
            pages_ = sorted({b.page for b in group})
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "source": section.source,
                        "page": pages_[0],
                        "pages": pages_,
                        "section": " > ".join(section.path),
                        "course_codes": course_codes(text),
                        "ocr": any(b.ocr for b in group),
                        "chunking": "structure",
                    },
                )
            )

        size = 0
        for b in blocks:
            if group and size + len(b.text) + 2 > limit:
                emit()
                group, size = [], 0
            group.append(b)
            size += len(b.text) + 2
        emit()

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "structure": structure_chunks}


def chunk_documents(pages: list[Document], cfg: Config) -> list[Document]:
    if cfg.chunking not in STRATEGIES:
        raise ValueError(f"Unknown chunking '{cfg.chunking}'. Choose from {list(STRATEGIES)}")
    return STRATEGIES[cfg.chunking](pages, cfg)
