# Task 2: Evaluation and error analysis

Applied Mathematics Major Handbook chatbot. All numbers come from `eval/results/summary.md`,
which is regenerated with `bash eval/run_all.sh` (about 1 h on an Apple M2 with 16 GB, Ollama on Metal).

## 1. Method

**Test set** (`questions.yaml`): 45 questions written from the handbook.

| Type | n | Expected behaviour |
|---|---|---|
| answerable | 38 | Correct facts with a page citation. Covers 10 categories: program requirements, minor, catalogue, course content, MLOs, tracks/careers, policies, flowchart (OCR page), multi-turn follow-ups |
| ambiguous | 4 | The handbook contradicts itself. The bot should report both versions |
| out_of_scope | 3 | Tuition, head of faculty, minimum GPA: not in the handbook. The bot should refuse |

Each question has a gold answer, gold PDF page(s), and 1–4 *key facts* (regular expressions).
The 3 multi-turn questions also store the conversation history and the expected standalone rewrite.

**Metrics** are measured at two levels, so that every wrong answer can be attributed to retrieval or to generation:

- *Retrieval*:
  - Hit@k and MRR of the gold page.
  - *Context recall*: the share of key facts that appear in the 5 retrieved chunks.
  - Page hits can be lucky (a different chunk on the same page). Context recall shows whether the evidence actually reached the LLM.
- *Answers*:
  - Labels: correct / incomplete / incorrect / false_refusal / correct_refusal / hallucination.
  - Score = (correct + correct_refusal + 0.5 × incomplete) / 45.

**Manual review.** Automatic labels come from regex matches, and they were wrong in 14 of the 135 answers of
the three main runs (10%). For example, the fallback prompt answered "a typical course is worth **3** credits",
and that answer was labelled *correct* because the regex matched "4 credits for seminars". All 135 answers were read,
and the corrections are stored in `manual_labels.yaml`. The report shows both the reviewed score and the automatic
score. The model-comparison runs (0.5b, 1.5b, llama) use automatic labels only.

## 2. Retrieval ablation (42 questions with gold pages, top-5)

| Parser | Chunking | Embedding | Retrieval | Context recall | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| pypdf | fixed | nomic | dense | 0.91 | 0.71 | 0.88 | 0.79 | ← baseline-like |
| pypdf | fixed | nomic | hybrid | 0.92 | 0.71 | 0.86 | 0.78 |
| pymupdf | fixed | nomic | hybrid | 0.91 | 0.74 | 0.88 | 0.80 |
| pymupdf | structure | nomic | dense | 0.90 | 0.67 | **0.98** | 0.81 |
| **pymupdf** | **structure** | **nomic** | **hybrid** | 0.92 | 0.83 | 0.95 | 0.88 | ← deployed |
| pypdf | structure | nomic | hybrid | 0.97 | 0.81 | 0.93 | 0.85 |
| pypdf | structure | mxbai | hybrid | **0.98** | **0.86** | **0.98** | **0.90** |

Findings:

1. **Structure-aware chunking is the largest single gain.** With the same parser, embedding and hybrid search,
   it raises Hit@1 from 0.71 to 0.81 (pypdf) and from 0.74 to 0.83 (pymupdf), and MRR by about 0.08.
   Each chunk is one course description or one requirement block, with its section path prepended,
   so the correct chunk ranks first instead of merely "somewhere in the top 5".
2. **Hybrid search mainly improves ranking.** With pymupdf + structure + nomic, hybrid raises Hit@1 from
   0.67 to 0.83 and MRR from 0.81 to 0.88. BM25 matches exact tokens such as "MATH 305" or "MLO 3",
   which the dense model treats as near-synonyms of their neighbours.
3. **pypdf + structure has a higher context recall than pymupdf + structure** (0.97 vs 0.92).
   The difference is 2–3 questions:
   - R1 ("128 credits") and R6 (General Curriculum): pypdf merges the end of page 9 and the start of page 10 into one chunk.
   - pymupdf splits them, and the chunk with the answer ranks 6th or lower.
   - This is a real effect but a small one on a 42-question set.
   
   The deployed default stays pymupdf. Its Markdown tables are what the table records (§3) are built from,
   and these records fixed most of the catalogue questions.
4. **mxbai-embed-large is best only with pypdf + structure + hybrid.** With pymupdf it is no better than nomic,
   and it is about 2× slower to embed. nomic-embed-text stays the default (smaller and faster). mxbai is a one-line switch (`EMBED_MODEL`).

## 3. Iteration: catalogue tables (round 1 → round 2)

The round-1 error review found that every catalogue question that failed had the same cause:

- The 16-row course catalogue table was **one chunk**, so its embedding was a blur of 16 courses.
- The rows continued on page 22 **without the header row**, so values lost their column names.
- The caption row "Courses in other majors cross-listed with Applied Mathematics" had been split by the parser
  into "c | ross-listed with Ap | plied", so BM25 could not match it.

Fix (`rag/chunking.py`, `linearize_table`):

- Each table row becomes a self-contained record, for example
  `Course Code: MATH 301 | Course Title: Optimization | Availability: Every other year | Prerequisites: MATH 101; MATH 103`.
- Records are packed at most 6 per chunk.
- The header is carried over to the next page, and the broken caption is re-joined.

| qwen2.5:3b strict, pymupdf/structure/hybrid | Hit@1 | Hit@5 | Catalogue Hit@5 | Catalogue answers fully correct |
|---|---|---|---|---|
| Round 1 (table = 1–2 chunks) | 0.76 | 0.93 | 0.63 | 4 / 8 |
| Round 2 (row records) | **0.83** | **0.95** | **0.75** | **5 / 8** |

- **Fixed:** C1 (MATH 301 offered every other year, now retrieved from the right row), C7 (ECON 209 and IS 312 cross-listed, previously "not found"), C5.
- **Regressed:** C4 ("How often is Introduction to Statistics offered?"). The row that says "Twice per year" now sits in a
  6-row chunk that ranks below the MATH 105 course description and other chunks.
- **Next step, not done:** append each course's catalogue record to its own description chunk, so that availability and
  prerequisites travel with the description. This would also put both versions of the conflicting prerequisites (A3, A4) in one chunk.

## 4. End-to-end answers (45 questions)

| Model | Prompt | Pipeline | Correct | Incompl. | Incorrect | False refusal | Correct refusal | Halluc. | **Score** | Median gen. time |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b | strict | pypdf / fixed / dense (baseline-like) | 19 | 8 | 8 | 7 | 3 | 0 | **0.58** | 13.0 s |
| qwen2.5:3b | strict | pymupdf / structure / hybrid | 27 | 6 | 6 | 3 | 3 | 0 | **0.73** | 13.7 s |
| qwen2.5:3b | fallback | pymupdf / structure / hybrid | 27 | 8 | 7 | 0 | 3 | 0 | **0.76** | 14.8 s |
| qwen2.5:1.5b* | strict | pymupdf / structure / hybrid | 28 | 6 | 8 | 0 | 3 | 0 | 0.76 | 6.2 s |
| llama3.2:3b* | strict | pymupdf / structure / hybrid | 27 | 6 | 7 | 2 | 3 | 0 | 0.73 | 17.0 s |
| qwen2.5:0.5b* | strict | pymupdf / structure / hybrid | 16 | 9 | 15 | 2 | 1 | 2 | 0.48 | 1.8 s |

\* automatic labels (not manually reviewed); expect about −0.03, as for the reviewed runs.

**Pipeline (same LLM, same prompt): 0.58 → 0.73.**
- The improved retrieval removes 4 of the 7 false refusals and 2 incorrect answers.
- The baseline-like pipeline refused 7 answerable questions:
  - For L4, T3 and P1, the evidence was missing or partial in its top 5 (context recall 0–0.67).
  - For C1, A3 and MT1, the evidence *was* retrieved, but inside long fixed-size chunks that mixed several topics, and the model refused anyway.
- It also answered C6 with the prerequisites of a neighbouring course, because a fixed-size chunk contained two course descriptions.

**Model size.**
- 0.5B is not usable: it is the only model that invents an answer to an out-of-scope question (2 hallucinations).
- 1.5B and 3B score the same here, but 1.5B is about 2× faster. It is a good choice for CPU-only Docker.
- llama3.2:3b is no better than qwen2.5:3b and is the slowest.
- The default stays qwen2.5:3b, the configuration that was fully reviewed. qwen2.5:1.5b is the recommended
  `LLM_MODEL` for CPU-only Docker. A manual review of the 1.5B run would be needed before calling it equivalent.

**Strict vs fallback prompt.** This result is surprising. The baseline "fallback" prompt (which allows general knowledge)
does not hallucinate on the three out-of-scope questions: with good retrieval it says the context has no
tuition or GPA information. Its total score is about the same as strict (0.76 vs 0.73), but it fails in a different way:

| | Strict | Fallback |
|---|---|---|
| False refusals (answer was in context) | 3 (R6, R7, C8) | 0 |
| Answers that **guess** beyond the context | 0 | 3: C8 "3 credits" (invented), C4 "likely every other year", T1 adds track descriptions not in the handbook |
| Out-of-scope refusals | 3 / 3, one exact sentence | 3 / 3, but long and with generic advice |

Strict is kept as the default. For a handbook assistant, *"I couldn't find this"* is a safe failure that the
student can act on, whereas a confident "3 credits" is a harmful one. The prompt-rule change from round 2
("never append the refusal to an answer; give the part you found") is in strict mode, and 2 of its 3 false refusals have
`context_recall = 0` (R6) or a gold page that was not retrieved (C8). Those two are really retrieval misses.

## 5. Error analysis (qwen2.5:3b, strict, deployed pipeline)

18 of the 45 answers are not fully correct. Each was traced to the stage where it went wrong:

| Root cause | Questions | Count |
|---|---|---|
| **Retrieval miss**: the evidence was not in the top 5 | R1, R6, C4, C8 (gold page 21 not retrieved), T3 (half of the list) | 5 |
| **OCR loses the diagram layout**: text is recovered but the boxes are not | F1, F2 | 2 |
| **Conflict not reported**: one version given, rule 4 ignored | A2, A3, A4 | 3 |
| **Generation error with the evidence in context** | C3, MT2, A1, R3 | 4 |
| **Over-cautious refusal with the evidence in context** | R7 | 1 |
| **Partially correct** (counted above) | R3, T3, F2, A2–A4 are *incomplete*, not wrong | – |

No answer was a hallucination: none of the wrong answers adds a fact that is absent from the retrieved excerpts. They misread, mix up or omit retrieved text.

### Example 1: multi-turn, correct with one prompt and incorrect with the other (MT2)
> **History:** "Tell me about the Real Analysis course." → *MATH 307 Real Analysis studies…*  **Q:** "Which course requires it as a prerequisite?"
> **Standalone query:** "Which course requires MATH 307 Real Analysis as a prerequisite?"

The rewrite resolves "it". With the fallback prompt, the answer is "MATH 312: Topology and Measure Theory requires MATH 307",
which is correct (p. 22 record, p. 28 description). With the strict prompt and the same retrieval, the model answered
"MATH 305 and MATH 311" (**incorrect**, MT2). The chunk contained several prerequisite lines, and the model picked the wrong ones.
This shows that small models make reading errors even when retrieval is perfect.

### Example 2: correct with citation (C7, fixed in round 2)
> **Q:** "Which courses from other majors are cross-listed with Applied Mathematics?"
> **A:** "ECON 209: Econometrics … IS 312: Biometry … (Source: p. 22)"

In round 1 this answer was "MATH 301: Optimization is cross-listed with CS", because the caption row was broken and the table was not retrieved.

### Example 3: incorrect, retrieval miss (R1)
> **Q:** "How many credits are required to graduate with a B.S. in Applied Mathematics?"
> **A:** "…the successful completion of **56** credits. (Source: p. 9)". Gold: **128** credits.

- Context recall is 0. The 128-credit sentence sits at the end of the *Elective Requirements* block (pages 9–10).
- The query matches the *Major Requirements (56 credits)* chunk much better.
- The model faithfully reports the wrong number it was given. The baseline-like pipeline makes the same mistake, even though there the right chunk *was* retrieved.
- A failure that is plausible but wrong like this is the strongest argument for showing sources in the UI.

### Example 4: incomplete, OCR of a diagram (F1 / F2)
> **Q:** "According to the flowchart, how many advanced courses are required?"
> **A:** "3 advanced courses". Gold: 2.

- Tesseract recovers the words of the image-only page 14, but it reads the flowchart boxes left-to-right across columns.
- As a result, "(2 required)" ends up next to "Intermediate or Advanced courses", and "Three main categories of areas" is adjacent to "ADVANCED COURSES".
- Every model and prompt fails F1. OCR solves *text* extraction, not *layout* understanding.
- Possible fixes: a hand-written alt-text description of the flowchart, or a vision-language model.

### Example 5: ambiguous, conflict not detected (A2)
> **Q:** "What is MATH 306?"
> **A:** "MATH 306 is titled 'Financial Mathematics'… cross-listed with ECON… prerequisite MATH 205."

- The course-description version (p. 27) is reported correctly.
- The alignment table (p. 17, also retrieved) lists MATH 306 as *Numerical Analysis*, and the model ignored it.
- Prompt rule 4 ("if excerpts contradict, report both") was not followed in any of the 4 ambiguous questions by any 3B model.
- Detecting a contradiction between two retrieved passages is beyond what these small models do reliably.

### Example 6: correct refusal (O1–O3)
> **Q:** "What is the tuition fee for the Applied Mathematics program?" → "I couldn't find this in the Applied Mathematics handbook."

All 3B and 1.5B runs refuse all 3 out-of-scope questions. qwen2.5:0.5b invents an answer for 2 of them.

## 6. Limitations of this evaluation

- 45 questions written by one person. Differences of 1–2 questions (about 0.03 in score) are within noise.
- Each configuration was run once. Generation at temperature 0.1 is not fully deterministic, and between
  rounds 1 and 2, R3 and MT2 changed label with no change in retrieval.
- Labels are regex-based plus one reviewer. The model-comparison runs were not manually reviewed.
- Latency was measured with Ollama on the Apple M2 GPU. CPU-only Docker is about 2.5–3× slower (38 s vs 14 s for one qwen2.5:3b answer).

## 7. Changes after the evaluation: follow-up questions and course lists

Found while testing the web UI. Conversation: "What are the requirements for a minor?" → "What are those
courses exactly?" → "Can you list them out?"

**Round A.** The bot repeated the same rule three times. There were two causes:
1. The generation prompt contained only the rewritten question, not the conversation.
2. The rewritten query retrieved the same passages again.

Fix: for follow-ups, add the last two exchanges and the original wording to the prompt, and retrieve 3 extra
passages. The bot then listed "Optimization, Statistical Learning, Numerical Analysis, …" as the minor's
courses. **That answer was wrong.** Those courses are the major's *Elective Applied Courses* list (p. 12).
The handbook names no fixed courses for the minor: any 6 MATH courses count, with at least three at the
200 level or above and at least one at the 300 level (p. 10).

**Round B: root cause in chunking.** The handbook marks sub-headings with italic lines (`_Sample Advanced
Courses_`, `_List of Applied Courses_`) and with label paragraphs ("Advanced Courses (300 level): …"). The
chunker did not detect either, so:
- the advanced-course list was filed under "Sample Foundation Courses";
- the elective list lost its "Elective Applied Courses" parent.

Fixes (`CHUNKER_VERSION = 3`):
- italic and label sub-headings are detected and nested. Section paths now read, for example,
  `… > Elective Applied Courses > List of Applied Courses > Applied Mathematics Courses` and
  `… > Advanced Courses (300 level) > Sample Advanced Courses`;
- catalogue records carry the handbook's level term, for example `Level: 300-level (Advanced)`;
- prompt rule 8: never present a list from one section as the answer for another; if only a rule exists, say so.

**Result.**
- The wrong course list is gone.
- On "what are those courses exactly?", qwen2.5:3b now restates the rule instead of saying that the handbook
  names no specific courses. That is not wrong, but it is not helpful.
- Two related limits remain:
  - **Retrieval.** Every rewrite of the follow-up keeps the word "minor", so the course catalogue is never
    retrieved. Multi-query rewriting (3 LLM-generated queries) was tried and did not help, because all 3 queries
    contained "minor".
  - **Reading tables.** In "Which sample advanced courses are offered every other year?" the catalogue records
    were retrieved, but the model listed MATH 308 (once a year) and missed several courses.

Re-running the 45 questions with these changes (automatic labels, qwen2.5:3b strict): context recall
0.917 → 0.917 (unchanged); 26 correct vs 27 before.
- **Gained:** F1 (flowchart: "at least 2 advanced courses").
- **Lost:** C8 (now guesses "3 credits") and P3 (one strategy missing).
- MT1–MT3 keep their labels.

At temperature 0.1, differences of one or two questions are within run-to-run noise. The tables in §2–§5
were produced before these changes.
