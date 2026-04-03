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

## Author

**Jamson Batista**  
GitHub: [@bawilla](https://github.com/bawilla)
