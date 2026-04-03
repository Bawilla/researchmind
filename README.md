# ResearchMind — Scientific Paper RAG System

ResearchMind is a retrieval-augmented generation (RAG) pipeline that lets you have an intelligent conversation with any scientific PDF. You drop in a paper, it chunks and embeds the content into a persistent vector store, then answers your questions by retrieving the most relevant passages and passing them as context to a large language model — giving you grounded, citation-aware answers instead of hallucinated summaries.

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Document loading & orchestration | LangChain + langchain-community |
| Vector store | ChromaDB (persisted locally) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | Groq — `llama-3.1-8b-instant` |
| PDF parsing | PyPDFLoader (pypdf) |

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/bawilla/researchmind.git
cd researchmind

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install langchain langchain-community langchain-text-splitters \
            chromadb sentence-transformers groq gradio pypdf
```

## Configuration

Set your Groq API key as an environment variable (get one free at console.groq.com):

```bash
# Windows
set GROQ_API_KEY=your_key_here

# macOS / Linux
export GROQ_API_KEY=your_key_here
```

Or create a `.env` file (it is gitignored):

```
GROQ_API_KEY=your_key_here
```

## Usage

1. Place a PDF in the project root named `paper.pdf`  
   (or download the RAG paper used in development):

```bash
curl -L -o paper.pdf "https://arxiv.org/pdf/2005.11401"
```

2. Run the interactive QA loop:

```bash
python main.py
```

On first run, the PDF is chunked, embedded, and stored in `./chroma_db/`.  
On subsequent runs, the existing store is loaded instantly — no re-indexing.

3. Type questions at the prompt. Type `quit` or `exit` to stop.

## Example Session

```
Loading existing ChromaDB from ./chroma_db...
  Loaded 158 chunks

ResearchMind QA — type 'exit' or 'quit' to stop.

Question: What is retrieval-augmented generation?

Answer:
RAG combines a pre-trained seq2seq model (parametric memory) with a dense
vector index of Wikipedia (non-parametric memory), accessed via a pre-trained
neural retriever. This allows the model to ground its generations in retrieved
evidence rather than relying solely on weights baked in during training.

[Retrieved 5 chunk(s) for this question]

Question: What datasets were used to evaluate RAG?

Answer:
The paper evaluates RAG on Natural Questions (NQ), TriviaQA, WebQuestions,
and CuratedTrec for open-domain QA, as well as MS-MARCO and Jeopardy question
generation tasks.

[Retrieved 5 chunk(s) for this question]
```

## Day 6 — RAGAS Evaluation Pipeline

`main6.py` runs an automated evaluation of the RAG pipeline using [RAGAS](https://docs.ragas.io/) metrics against 10 quantum error correction questions with known reference answers.

### Setup

```bash
pip install ragas datasets langchain-groq
```

### Run

```bash
python main6.py
```

Results are saved to `evaluation_results.json`.

### Evaluation Results

**Model:** `llama-3.1-8b-instant` | **Papers:** 5 QEC papers (1 804 chunks) | **Retrieval:** top-10 → rerank → top-3

| Question | Faithfulness | Relevancy | Context Recall |
|---|---|---|---|
| What is quantum error correction? | — | 1.000 | — |
| What is a stabilizer code? | — | 1.000 | — |
| How does the surface code correct errors? | — | 0.789 | 0.500 |
| What is the threshold error rate…? | — | 0.000 | — |
| What is the distance of a QEC code? | 0.667 | — | — |
| What is the toric code? | 0.750 | — | — |
| What are the main types of quantum errors? | 0.714 | — | 0.000 |
| What is fault-tolerant quantum computing? | 1.000 | — | 0.333 |
| How does MWPM work in QEC? | 0.250 | — | 0.333 |
| How does magic state distillation…? | 0.167 | — | 1.000 |
| **Average** | **0.5913** | **0.6972** | **0.4332** |

> Note: Some cells show `—` where the Groq API returned partial results due to rate-limit constraints (`n > 1` not supported). Averages are computed over the available non-null scores.

## Author

**Jamson Batista**  
GitHub: [@bawilla](https://github.com/bawilla)
