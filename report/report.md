---
title: "A Local RAG Assistant for the FUV Applied Mathematics Major Handbook"
subtitle: "CS311 Advanced Deep Learning, Fall 2026: Final Project Report"
author: "STUDENT_NAME (STUDENT_ID)"
date: "November 2026"
---

# 1. Introduction

Students who want to know how many credits a minor needs, which course is a prerequisite for Stochastic
Calculus, or whether the Capstone is mandatory usually have to search a 29-page PDF handbook. This project
builds a chatbot that answers such questions from the **B.S. Applied Mathematics Major Handbook** of Fulbright
University Vietnam. Every answer cites the handbook pages it comes from.

The system is a **fully local Retrieval-Augmented Generation (RAG)** pipeline:

- documents are parsed, indexed and searched on the student's own machine;
- a small open model (Qwen 2.5, 3B parameters) served by Ollama writes the answer;
- no text is sent to a cloud API.

The work started from the course's baseline script (`run_txt.py`) and extended it in four directions, which
match the project tasks:

1. PDF support with OCR for image-only pages (Task 1).
2. A handbook chatbot that was evaluated systematically and improved over three rounds (Task 2).
3. A reproducible Docker deployment (Task 3).
4. A web interface designed around evidence and citations (Task 4).

This report describes the design, the implementation and the evaluation, and it discusses what worked, what failed
and why. All numbers can be reproduced with the commands in the appendix. The full per-question results are in
`source_code/eval/`.

# 2. Starting point

## 2.1 The baseline

The baseline loads `.txt` files and splits them into 1,000-character chunks. It embeds the chunks with
`nomic-embed-text`, stores them in Chroma, and asks `qwen2.5:0.5b` (temperature 0.7) to answer. Its prompt
explicitly allows the model to "use your general knowledge" when the context is insufficient. For a handbook
assistant, each of these choices is a problem:

| Baseline choice | Problem for this use case |
|---|---|
| `.txt` loader only | The handbook is a PDF. Page 14 (the program flowchart) is an image with no text layer |
| Fixed 1,000-character chunks | Chunks cut through course descriptions and tables and carry no section information |
| Dense-only retrieval | Exact tokens such as "MATH 305" or "MLO 3" are matched poorly |
| "Use general knowledge" prompt | Invites invented credits and policies, which is the worst failure for an advising tool |
| 0.5B model, temperature 0.7 | Too small and too random for factual questions |
| Re-index only when the folder is missing | Changing a document silently keeps a stale index |

## 2.2 The knowledge source

The handbook has about 7,700 words on 29 pages and is a mixture of content types:

- prose (mission, teaching philosophy);
- numbered requirement lists (credits for the major and minor);
- tables (the course catalogue, and a course × learning-outcome matrix);
- a year-by-year study plan laid out as a grid;
- a full-page flowchart image.

Reading it closely also revealed that **the handbook contradicts itself** in several places:

- MATH 306 is "Financial Mathematics" in the course descriptions but "Numerical Analysis" in the
  outcome-alignment table.
- One section says at least one elective applied course must be at the 200 level, while two others say 300.
- The catalogue table and the course descriptions list different prerequisites for MATH 308 and MATH 310.

These cases were kept as a separate "ambiguous" category in the test set, because a trustworthy assistant should
report both versions instead of choosing one.

# 3. System architecture

```
 Indexing (on start-up, and whenever a document or a setting changes)
 PDF ─► PyMuPDF / pymupdf4llm ─► clean ─► Tesseract OCR (image-only pages) ─► structure-aware chunker
     ─► nomic-embed-text (dense, 768-d) + BM25 (sparse) ─► Qdrant collection (+ corpus hash)

 Question answering (per request)
 question + history ─► query rewriting (LLM) ─► hybrid search (dense + BM25, RRF) ─► top-k passages
     ─► grounded prompt with page-numbered excerpts ─► qwen2.5:3b (Ollama) ─► answer + citations
     ─► FastAPI (REST + Server-Sent Events) ─► web UI (nginx)
```

The code is a small Python package (`source_code/rag/`) with one module per stage: `loaders`, `chunking`,
`index`, `prompts` and `chain`. A YAML configuration file controls the pipeline, and every key can be overridden
by an environment variable. The same code therefore runs from the command line, inside the evaluation scripts,
behind the API and in Docker.

**Vector database.** Qdrant was chosen over Chroma for three reasons:

- It stores a dense vector and a sparse (BM25) vector for each chunk and fuses the two result lists with
  Reciprocal Rank Fusion (RRF) inside the database.
- It runs as a separate container with a dashboard, which suits the Docker task.
- It has an embedded mode for development.

Each collection name encodes the parser, chunking strategy and embedding model, so different experimental
configurations never overwrite each other. Every point also stores a hash of the documents and of the chunking
settings. When either changes, the index is rebuilt automatically. The baseline only re-indexed when its folder
was deleted.

# 4. Implementation

## 4.1 PDF processing and OCR (Task 1)

Two parsers are available behind one switch:

- `pypdf` (plain text, close to the assignment's suggestion);
- `pymupdf4llm`, which converts each page to Markdown and keeps headings and tables.

After parsing, a cleaning step does three things:

- removes table-of-contents dot leaders, page numbers and HTML remnants;
- repairs ligatures;
- drops orphan table rules.

Each page keeps its 1-based page number as metadata, so every chunk can be cited.

Any page whose text layer has fewer than 50 characters is treated as an image. It is rendered at 300 DPI and read
with Tesseract. This recovers the flowchart on page 14, which the baseline could not see at all. The built-in OCR
of pymupdf4llm was tried first, but it produced noticeably noisier text than calling Tesseract directly.

## 4.2 Structure-aware chunking

Fixed-size chunking was the baseline's main weakness, and replacing it gave the largest single improvement
(Section 5.2). The structure chunker:

1. **Detects headings** from Markdown markers, numbered titles ("5.1."), lines matching table-of-contents
   entries, italic sub-headings ("_Sample Advanced Courses_") and label paragraphs ("Advanced Courses
   (300 level): …").
2. **Builds a section tree**, so that every chunk starts with its path, for example
   `Section: 6. Course Catalogue > Course Descriptions > MATH 305: Stochastic Calculus`. The embedding and BM25
   then both "know" where the text comes from.
3. **Gives each course description its own chunk**, and never lets a chunk span a page boundary within a block,
   so citations stay exact.
4. **Linearises tables into records.** Each catalogue row becomes a self-contained line, for example
   `Course Code: MATH 301 | Course Title: Optimization | Level: 300-level (Advanced) | Availability: Every
   other year | Prerequisites: MATH 101; MATH 103`. Records are packed at most six per chunk. A table that
   continues on the next page reuses the previous header, and caption rows that the parser split across cells
   are re-joined.

Points 1 (the italic and label sub-headings) and 4 were added in later rounds, as a direct result of the error
analysis (Section 5.3).

## 4.3 Retrieval

Two models embed each chunk:

- **Dense:** `nomic-embed-text`, with the task prefixes the model was trained with (`search_document:` and
  `search_query:`). The baseline omitted them.
- **Sparse:** BM25 (fastembed's `Qdrant/bm25` model, with IDF computed by Qdrant).

Dense search finds paraphrases ("how often is it offered" ↔ "Availability: Every other year"). BM25 finds exact
identifiers that dense models blur together ("MATH 305" vs "MATH 308"). Qdrant fuses the two ranked lists with
RRF, which uses ranks rather than raw scores, so the two incompatible score scales do not need to be calibrated.
The top 5 passages are passed to the model.

## 4.4 Generation and prompt design

The **strict** prompt gives the model eight rules:

- use only the excerpts;
- if nothing is relevant, reply with one fixed sentence ("I couldn't find this in the Applied Mathematics
  handbook.");
- copy numbers and course codes exactly;
- report both versions when excerpts contradict each other;
- end with the cited pages;
- be concise;
- for a follow-up, add only new information;
- never present a list from one section as the answer for another.

Each excerpt is labelled with its page number, so the model can cite pages. The baseline prompt was kept as a
**fallback** mode for comparison.

**Conversations.** A follow-up such as "What are its prerequisites?" cannot be searched as it is. The question is
first rewritten by the LLM into a standalone query ("What are the prerequisites for MATH 205?"), and that query is
used for retrieval. For follow-ups, the last two exchanges are also shown to the model, and three extra passages
are retrieved.

One small post-processing guard exists because 3B models sometimes answer correctly and then append the
refusal sentence anyway: a trailing refusal is removed when a real answer precedes it.

Generation runs at temperature 0.1 with an 8,192-token context.

# 5. Evaluation (Task 2)

## 5.1 Method

The test set has 45 questions written from the handbook:

| Type | n | Expected behaviour |
|---|---|---|
| Answerable | 38 | Correct facts with page citations. Covers requirements, the minor, the catalogue, course content, learning outcomes, careers, policies, the flowchart page, and 3 multi-turn follow-ups |
| Ambiguous | 4 | The handbook contradicts itself; both versions should be reported |
| Out of scope | 3 | Tuition, head of faculty, minimum GPA; the bot must refuse |

Each question has a gold answer, gold pages and 1–4 *key facts* written as regular expressions.

**Measuring at two levels.** Every failure was attributed to either retrieval or generation:

- **Retrieval:** Hit@k and MRR of the gold page, plus **context recall**, the share of key facts present in the
  retrieved text. Page hits can be lucky, because a different chunk on the right page also counts as a hit;
  context recall shows whether the evidence actually reached the model.
- **Answers:** each answer is labelled *correct*, *incomplete*, *incorrect*, *false refusal*, *correct refusal* or
  *hallucination*. The score is (correct + correct refusal + 0.5 × incomplete) / 45.

**Manual review.** Labels are first assigned automatically from the key facts. The automatic labels turned out to
be wrong for 14 of 135 answers (10%). For example, an answer claiming courses are worth "3 credits" was marked
correct because the regex for "4" matched "4 credits for seminars". Every answer of the three main runs was
therefore read and corrected by hand.

## 5.2 Results

**Retrieval ablation** (42 questions with gold pages, top 5; round 2):

| Parser | Chunking | Retrieval | Context recall | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|---|
| pypdf | fixed | dense\* | 0.91 | 0.71 | 0.88 | 0.79 |
| pymupdf | fixed | hybrid | 0.91 | 0.74 | 0.88 | 0.80 |
| pymupdf | structure | dense | 0.90 | 0.67 | 0.98 | 0.81 |
| **pymupdf** | **structure** | **hybrid** | 0.92 | **0.83** | 0.95 | **0.88** |
| pypdf | structure | hybrid | 0.97 | 0.81 | 0.93 | 0.85 |

\* closest to the baseline. Bold: the deployed configuration.

What the ablation shows:

- **Structure-aware chunking** is the largest gain: about +0.10 Hit@1 for both parsers.
- **Hybrid search** mainly improves ranking: Hit@1 rises from 0.67 to 0.83 with the same chunks.
- **pypdf + structure has a slightly higher context recall** (0.97 vs 0.92). On a 42-question set this is
  2–3 questions, caused by how the two parsers split pages 9–10.
- Replacing `nomic-embed-text` with `mxbai-embed-large` helped only in one configuration and is twice as slow.

**End-to-end answers** (45 questions, manually reviewed; round 2):

| Run | Correct | Incompl. | Incorrect | False refusal | Correct refusal | Score |
|------------------------------|---------|---------|---------|----------|----------|-------|
| A. Baseline-like, strict | 19 | 8 | 8 | 7 | 3 | 0.58 |
| B. Improved, strict | 27 | 6 | 6 | 3 | 3 | **0.73** |
| C. Improved, fallback | 27 | 8 | 7 | 0 | 3 | 0.76 |
| D. Deployed (round 3), strict | 29 | 5 | 6 | 2 | 3 | **0.77** |

A uses pypdf, fixed chunks and dense retrieval. B, C and D use pymupdf, structure chunking and hybrid
retrieval. A–C are from round 2; D includes the round-3 changes. All four use qwen2.5:3b.

The improved pipeline raises the score from **0.58 to 0.73** with the same model and the same prompt. Better
retrieval alone removes four of the seven false refusals: the baseline-like pipeline often refused because the
right passage was not in its top 5.

After the round-3 changes (Section 5.3), the deployed system scores **0.77** (29 correct, no hallucinations),
with a median generation time of 5.6 s. Its retrieval metrics changed only slightly:

- context recall 0.92 (unchanged);
- Hit@5 0.95 (unchanged);
- Hit@1 0.76 (down from 0.83), because the new, smaller list chunks compete with the requirement chunks for the
  first rank.

The flowchart questions F1 and F2, wrong in every earlier run, are now answered correctly. The other comparisons
in this section were measured in round 2 and were not re-run.

**Language models** (same retrieval, strict prompt, automatic labels):

| Model | Score | Median generation time |
|---|---|---|
| qwen2.5:0.5b | 0.48 | 1.8 s |
| qwen2.5:1.5b | 0.76 | 6.2 s |
| qwen2.5:3b | 0.76 (0.73 after review) | 13.7 s |
| llama3.2:3b | 0.73 | 17.0 s |

- 0.5B is the only model that invents answers to out-of-scope questions (2 of 3).
- 1.5B and 3B score alike, and 1.5B is twice as fast, which makes it a good choice for CPU-only Docker.
- qwen2.5:3b remains the default because it is the configuration that was fully reviewed.

(Times were measured while the evaluation was saturating the machine. In normal use, answers take about
4–10 s on the M2 GPU.)

**Strict vs fallback prompt.** With good retrieval, the baseline prompt did *not* hallucinate on the
out-of-scope questions, and its score is similar to strict. The difference is *how* they fail:

- **Strict** failed by refusing three times when the answer was present.
- **Fallback** never refused, but it guessed three times: it invented "3 credits", answered "likely every other
  year" for a course offered twice per year, and decorated an answer with track descriptions that are not in
  the handbook.

For an advising tool, "I couldn't find this" is a safe failure that the student can act on, while a confident
wrong number is harmful. Strict mode was therefore kept as the default.

## 5.3 Error analysis and iterations

The deployed configuration got 18 of 45 answers not fully correct in round 2. Their causes:

| Root cause | Questions |
|---|---|
| Retrieval miss (evidence not in the top 5) | R1 "128 credits", R6, C4, C8, T3 (partly) |
| OCR loses the flowchart's layout | F1, F2 |
| Contradiction not reported (rule 4 ignored) | A2, A3, A4 |
| Generation error although the evidence was retrieved | C3, MT2, A1, R3 |
| Over-cautious refusal | R7 |

No answer contained a fact that was absent from its excerpts. The failures are misreadings, mix-ups and
omissions.

**Example: a plausible wrong answer.** Asked how many credits the degree requires, the bot answered
"**56** credits (p. 9)". The correct figure is 128. The 128-credit sentence sits at the end of the
*Elective Requirements* block, while the query matches the *Major Requirements (56 credits)* chunk much better.
The model faithfully reported the wrong number it was given. This failure is the strongest argument for showing
the sources next to every answer (Section 7). It is still wrong in round 3.

**Example: OCR is not layout understanding.** Tesseract recovers the words on the flowchart page but reads the
boxes left to right across columns, so "(2 required)" lands next to the wrong box. Every model answered
"3 advanced courses" instead of 2 in rounds 1 and 2. In round 3 both flowchart questions were answered
correctly, probably because the surrounding sections now carry clearer headings. This is one run, so it may not
be stable.

**Example: contradictions.** For "What is MATH 306?" the model reported only "Financial Mathematics", although
the excerpt calling it "Numerical Analysis" was also retrieved. No 3B model followed the "report both versions"
rule on any of the four ambiguous questions.

The errors led to three rounds of changes:

- **Round 1 → 2: tables.** All failing catalogue questions had one cause. The 16-row catalogue table was a single
  chunk with a blurred embedding, its continuation on page 22 had no header, and a caption row was split into
  "c | ross-listed with Ap | plied". Linearising rows into records raised Hit@1 from 0.76 to 0.83 and fixed C1 and
  C7. One question (C4) regressed.
- **Round 2 → 3: follow-ups and course lists.** This round came from manual testing of the UI rather than from
  the test set. In "What are the requirements for a minor?" → "What are those courses exactly?" the bot first
  repeated the rule. After the conversation was added to the prompt, it listed courses, but they were the wrong
  ones: it presented the major's *elective applied* list as the minor's courses. The handbook names no fixed
  courses for the minor (any six MATH courses with level constraints).

  The root cause was in chunking. The handbook marks sub-sections with italic lines and "X Courses (N level):"
  label paragraphs, which were not detected, so the elective list had lost its "Elective Applied Courses" parent.
  Detecting these headings, adding the course level to catalogue records, and adding prompt rule 8 removed the
  wrong list. The 3B model still restates the rule instead of saying "the handbook names no specific courses".
  This is a limitation (Section 9). On the test set, round 3 scores 0.77 vs 0.73. The remaining 13 answers that
  are not fully correct have the same causes as in the table above, minus the flowchart.

# 6. Docker deployment (Task 3)

`docker/docker-compose.yml` starts five services in dependency order, using health checks:

| Service | Role |
|---|---|
| `ollama` | LLM and embedding server; models are kept in a named volume |
| `ollama-init` | One-shot job that pulls the required models, then exits |
| `qdrant` | Vector database (port 6333, with a dashboard); data in a named volume |
| `api` | FastAPI + RAG core (port 8000). Builds or verifies the index on start |
| `ui` | nginx (port 8080). Serves the web UI and proxies `/api` to FastAPI |

Design details:

- **API image** (1.06 GB): `python:3.11-slim`, Tesseract, a non-root user and a health check. The BM25 model is
  baked in at build time, so the container works offline.
- **UI image:** plain nginx, with no Python at all.
- **Documents** are mounted read-only.
- **Restarts:** because of the corpus hash, a restart only re-indexes when something changed.

**Apple Silicon.** Docker on macOS cannot use the Metal GPU, so the bundled `ollama` container runs on CPU.
Measured on the M2, one answer took about 38 s, against about 14 s with Ollama running natively. An override file
(`docker-compose.host-ollama.yml`) therefore disables the bundled Ollama and points the API at the host's Ollama.
This gives one command for graders (fully containerised) and a fast mode for demos. Both modes were tested end to
end.

# 7. User interface (Task 4)

The first interface was written in Streamlit and worked. It was replaced because its layout could not do what
testing showed to be most important for this domain: **putting the answer and its evidence side by side**.

The R1 example ("56 credits") is exactly the kind of answer a student would believe, unless the cited passage is
one glance away.

The final UI is a static web app (HTML, CSS and JavaScript, with no build step) with a black and ocean-blue
palette and the IBM Plex typefaces:

- **Answers** stream token by token, in a serif face for readability.
- **Page citations** such as `p. 22` are clickable chips. A click highlights the matching passage in the
  **Sources** panel, where each passage shows its page, section path, relevance score, an OCR flag and the full
  text.
- **The Handbook tab** shows the cited page itself, rendered by the API with PyMuPDF. Chrome's embedded PDF viewer
  was tried first and dropped: in a narrow panel it zoomed to the wrong region and showed browser pop-ups.
- **Follow-ups** show "Searched for: …", the rewritten query, so the user can see how a question was interpreted.
- **A refusal** looks like a neutral note, not an error.
- **Settings:** model, number of passages, temperature, strict/fallback mode and light/dark appearance.
- **Status:** the LLM and the index are shown as up or down, with the chunk count and a "Rebuild index" button.

The fonts are bundled, so the UI works offline. The layout adapts to phone width.

# 8. Observations and lessons learned

1. **Retrieval quality dominates.** The same 3B model went from 0.58 to 0.73 by changing only how the document
   was parsed, chunked and searched. Most "LLM errors" in early testing were really retrieval misses.
2. **Document structure is data.** The biggest fixes were about recovering structure the PDF hides: table
   headers on the next page, captions split across cells, italic sub-headings. A chunk without its section path
   is easy to misread: the elective-course list without its heading looked like "the courses".
3. **Measure at two levels.** Context recall separated "the evidence never arrived" from "the model misread it".
   Without it, R1 and MT2 look like the same kind of failure, but they need opposite fixes.
4. **Automatic metrics need a human check.** Regex labels were wrong 10% of the time, in both directions.
5. **Testing like a user finds different bugs.** The worst error (a wrong course list, presented with
   confidence) did not appear in the 45-question set at all. It appeared in a three-turn conversation in the UI.
6. **Small models follow simple rules but not subtle ones.** They reliably refuse out-of-scope questions and
   copy numbers, but they rarely report contradictions or say "the handbook has a rule, not a list".
7. **Prompt wording position matters.** An instruction placed in the system prompt was ignored by the 3B model.
   The same instruction placed right after the question worked. However, one over-cautious sentence turned a
   correct follow-up answer into a refusal, so every prompt change had to be re-tested on the whole set.

# 9. Limitations

- **Model capacity.** The 3B model misreads tables (for example, it listed MATH 308 among courses offered every
  other year), ignores the contradiction rule, and restates rules instead of explaining what they allow.
- **Retrieval for broad follow-ups.** A rewritten follow-up keeps the topic word ("minor"), so passages
  elsewhere in the handbook (the catalogue) are not retrieved. Multi-query rewriting was tried and did not help,
  because every generated query kept the same word.
- **OCR of diagrams.** The flowchart's text is recovered, but not its structure.
- **Evaluation size.** 45 questions written by one person, each configuration run once at temperature 0.1.
  Differences of one or two questions are within run-to-run noise. The model-comparison runs were not
  manually reviewed.
- **CPU-only Docker on macOS** is about 2.7× slower than native Ollama.

# 10. Future improvements

- **A larger local model** (qwen2.5:7b fits in 16 GB), compared on the same test set.
- **Fine-tuning with LoRA/QLoRA** (for example with `mlx-lm` on Apple Silicon) on question + excerpt → cited-answer
  pairs. The goal would be to teach behaviour such as reporting contradictions and distinguishing rules from lists,
  with RAG still supplying the facts. The test set must stay out of the training data.
- **Enrich course descriptions with their catalogue record**, so that availability and prerequisites travel
  with the description. This would also place both versions of contradictory prerequisites in one chunk.
- **A reranker** (a small cross-encoder) over the top 20 hybrid results.
- **Structured lookups** for questions that are really database queries ("which courses are offered every other
  year?"): answer them from the parsed catalogue table instead of asking the LLM to scan records.
- **A hand-written text description of the flowchart**, or a vision-language model, for page 14.

# Appendix: reproducing the results

```bash
# local development (Python 3.11, Ollama and Tesseract installed)
cd source_code
pip install -r requirements.txt
ollama pull qwen2.5:3b && ollama pull nomic-embed-text
python -m rag.cli index
python -m rag.cli ask -q "What are the prerequisites for MATH 305?"
uvicorn api.main:app --port 8000            # API docs at /docs, web UI at /ui/

# evaluation
bash eval/run_all.sh                         # writes eval/results/summary.md

# Docker
cd docker
docker compose up --build -d                 # fully containerised, UI at http://localhost:8080
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build -d   # host Ollama (GPU)
```
