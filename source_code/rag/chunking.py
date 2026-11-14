"""Chunking strategies.

* ``fixed``     - baseline: RecursiveCharacterTextSplitter (chunk_size / chunk_overlap), applied
                  per page so every chunk keeps an exact page number.
* ``structure`` - heading-aware: the Markdown produced by pymupdf4llm is split into sections
                  (e.g. "4. Program Structure and Curriculum > Minor Requirements (24 credits)"),
                  each course description becomes its own chunk, tables are never cut in the
                  middle of a row, and the section path is prepended to the chunk text so that
                  the embedding "knows" where the text comes from.

Tables (structure chunking) are linearised into one self-contained record per row
("Course Code: MATH 301 | Course Title: Optimization | Availability: Every other year | ...")
and packed into chunks of at most ``table_rows`` records. This fixes three problems seen in
evaluation round 1: a 16-row catalogue table in one chunk gave a diluted embedding, a table
continued on the next page lost its header row, and caption rows that the PDF parser broke
across cells ("Courses in other majors c|ross-listed with Ap|plied Mathematics") were invisible
to BM25.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config

CHUNKER_VERSION = 3   # bump when chunk output changes -> triggers re-indexing

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
_ITALIC_HEADING = re.compile(r"^_([A-Z][A-Za-z ,&/()-]{2,60}[A-Za-z)])_$")   # "_Sample Advanced Courses_"
# "Advanced Courses (300 level): ...", "Research and Writing Seminar: ...", "Capstone Project (optional): ..."
_COURSE_LABEL = re.compile(r"^([A-Z][A-Za-z ]{2,40} (?:Courses|Seminar|Project)(?: \([\w ]{3,20}\))?):\s+\S")
ITALIC = 0   # pseudo-level returned by _detect_heading for italic sub-headings
_COURSE_LEVELS = {"1": "100-level (Foundation)", "2": "200-level (Intermediate)",
                  "3": "300-level (Advanced)", "4": "400-level (Capstone)"}
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
    table_header: list[str] | None = None   # carried over when a table continues on the next page


_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")


def _cells(row: str) -> list[str]:
    row = row.strip()
    row = row[1:] if row.startswith("|") else row      # remove only the outer pipes so that
    row = row[:-1] if row.endswith("|") else row       # empty first/last cells are kept
    return [c.strip() for c in row.split("|")]


def _clean_header(cell: str) -> str:
    cell = re.sub(r"-\s+", "-", cell).strip()          # "CROSS- LISTING" -> "CROSS-LISTING"
    return " ".join(w.capitalize() if w.replace("-", "").isalpha() and w.isupper() and len(w) > 3 else w
                    for w in cell.split())               # "COURSE CODE" -> "Course Code", "MLO1" kept


def linearize_table(text: str, section: _Section) -> str:
    """Markdown table -> one 'Header: value | Header: value' record per line.

    * the row before the |---| rule is the header, unless the section already had a table with
      the same number of columns and a different header (= continuation on a new page);
    * a row with an empty first cell and the rest of its text split over several cells is a
      caption that the parser broke apart; its cells are re-joined and it prefixes the following
      records."""
    rows = [r for r in text.splitlines() if r.strip()]
    body_rows = [r for r in rows if not _TABLE_RULE.match(r.strip().strip("|").strip())]
    if not body_rows:
        return text
    first = _cells(body_rows[0])
    prev = section.table_header
    if prev and len(prev) == len(first) and [_clean_header(c) for c in first] != prev:
        header, data = prev, body_rows                  # continuation: first row is data
    else:
        header, data = [_clean_header(c) for c in first], body_rows[1:]
    section.table_header = header

    records, caption = [], ""
    for row in data:
        cells = _cells(row)
        filled = [c for c in cells if c]
        if not filled:
            continue
        broken = any(a[-1:].islower() and b[:1].islower() for a, b in zip(filled, filled[1:]))
        if not cells[0] and len(filled) >= 2 and broken:
            caption = "".join(filled)                    # broken-up caption row
            records.append(caption)
            continue
        fields = [f"{h}: {c}" if h else c for h, c in zip(header, cells) if c]
        code = COURSE_CODE.fullmatch(cells[0]) if cells and header and "code" in header[0].lower() else None
        if code and code.group(1) == "MATH" and code.group(2)[0] in _COURSE_LEVELS:   # e.g. "300-level (Advanced)"
            fields.insert(2, f"Level: {_COURSE_LEVELS[code.group(2)[0]]}")
        fields += [c for c in cells[len(header):] if c]
        records.append((f"[{caption}] " if caption else "") + " | ".join(fields))
    return "\n".join(records)


def _detect_heading(line: str, toc: set[str]) -> tuple[int, str] | None:
    """Return (level, title) if the line is a section heading.

    Four signals: Markdown '#' headings from the parser, numbered headings ("5.1. ..."), plain lines
    that exactly match a table-of-contents entry, and short italic-only lines ("_Sample Advanced
    Courses_"), which the handbook uses as sub-headings without a heading font."""
    s = line.strip()
    m = _MD_HEADING.match(s)
    if m and not _ITALIC_HEADING.match(m.group(2).strip()):
        title = m.group(2).strip(" _*:")
        if not title or _NOT_A_HEADING.match(title):
            return None
        return _level_from_number(title, len(m.group(1))), title
    m = _ITALIC_HEADING.match(m.group(2).strip() if m else s)
    if m and len(m.group(1).split()) <= 6:
        return ITALIC, m.group(1)     # level resolved by _build_sections from the context
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

    # Handbook-specific nesting below a numbered/### section (level `top`):
    #   "X Courses (N level): description" paragraph   -> label sub-section       (top + 1)
    #   "_Sample X Courses_" italic line after a label -> child of the label       (top + 2)
    #   "_List of Applied Courses_" italic line         -> sub-section that owns the
    #   "#### Economics Courses" ... headings that follow it                        (+1)
    top, label_active, list_level = 3, False, None
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
            if text and para_is_table:
                text = linearize_table(text, current)
            if text:
                current.blocks.append(_Block(text, page, ocr, para_is_table))
            para, para_is_table = [], False

        for line in lines:
            heading = None if ocr else _detect_heading(line, toc)
            label = None if ocr or heading else _COURSE_LABEL.match(line.strip())
            if label:                   # "Advanced Courses (300 level): ..." starts a sub-section
                heading = (top + 1, label.group(1))
                label_active, list_level = True, None
            elif heading:
                level, title = heading
                if level == ITALIC:
                    level = top + 2 if label_active else top + 1
                    list_level = level if title.lower().startswith("list of") else None
                elif list_level is not None and level == top + 1:
                    level = list_level + 1          # e.g. "Economics Courses" under the list
                elif level > 1:
                    label_active, list_level = False, None
                    if level <= 3:
                        top = level
                heading = (level, title)
            if heading:
                flush()
                level, title = heading
                if level == 1:          # document title ("Undergraduate Major Handbook")
                    continue
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                start(source)
                if label:               # keep the description paragraph as the section's text
                    para.append(line)
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


def _split_table(block: _Block, limit: int, max_rows: int) -> list[_Block]:
    """Split linearised table records into groups (records are self-contained, so no header)."""
    out, cur = [], []
    for row in block.text.splitlines():
        if cur and (len(cur) >= max_rows or len("\n".join(cur + [row])) > limit):
            out.append(_Block("\n".join(cur), block.page, block.ocr, True))
            cur = []
        cur.append(row)
    if cur:
        out.append(_Block("\n".join(cur), block.page, block.ocr, True))
    return out


def _split_long(block: _Block, cfg: Config, limit: int) -> list[_Block]:
    if block.is_table:
        return _split_table(block, limit, cfg.table_rows)
    if len(block.text) <= limit:
        return [block]
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
            if b.is_table:                  # table record groups are always chunks of their own
                emit()
                group, size = [b], 0
                emit()
                group = []
                continue
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
