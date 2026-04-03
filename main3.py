import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from groq import Groq

PAPERS_DIR = "./papers"
CHROMA_DIR = "./chroma_db3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
RETRIEVAL_K = 10   # candidates fetched from ChromaDB
RERANK_TOP_N = 3   # chunks passed to LLM after reranking


def infer_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def load_all_papers(papers_dir: str) -> list:
    pdf_files = sorted(Path(papers_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  Loading {pdf_path.name} ...", end=" ", flush=True)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["title"] = infer_title(pdf_path.name)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"{len(pages)} pages → {len(chunks)} chunks")

    return all_chunks


def build_or_load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing ChromaDB from {CHROMA_DIR}...")
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        print(f"  Loaded {vs._collection.count()} chunks\n")
        return vs

    print(f"Indexing all PDFs in {PAPERS_DIR}/...")
    chunks = load_all_papers(PAPERS_DIR)
    print(f"\nEmbedding {len(chunks)} total chunks into {CHROMA_DIR}/ ...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done — {vs._collection.count()} chunks stored.\n")
    return vs


# ---------------------------------------------------------------------------
# Metadata filter parsing (identical to main2.py)
# ---------------------------------------------------------------------------
_FILTER_PATTERNS = [
    r"what does ([a-zA-Z0-9_\-]+\.pdf) say about (.+)",
    r"according to ([a-zA-Z0-9_\-]+\.pdf)[,\s]+(.+)",
    r"in ([a-zA-Z0-9_\-]+\.pdf)[,\s]+(.+)",
]


def parse_filter(question: str):
    for pattern in _FILTER_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            return m.group(2).strip("?. "), {"source": m.group(1)}
    return question, None


# ---------------------------------------------------------------------------
# Core QA with reranking
# ---------------------------------------------------------------------------
def answer_question(
    client: Groq,
    vectorstore: Chroma,
    reranker: CrossEncoder,
    question: str,
) -> None:
    clean_q, where_filter = parse_filter(question)

    # Stage 1 — retrieve candidates from ChromaDB
    if where_filter:
        candidates = vectorstore.similarity_search(
            clean_q, k=RETRIEVAL_K, filter=where_filter
        )
        filter_note = f" | filtered to {where_filter['source']}"
    else:
        candidates = vectorstore.similarity_search(clean_q, k=RETRIEVAL_K)
        filter_note = ""

    if not candidates:
        print("\n[No candidates found — try a different question or filename.]\n")
        return

    # Stage 2 — rerank with cross-encoder
    pairs = [[clean_q, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:RERANK_TOP_N]]
    top_scores = [score for score, _ in scored[:RERANK_TOP_N]]

    # Stage 3 — build context with citations
    context_parts = []
    for doc in top_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{src}, p.{page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # Stage 4 — generate answer
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant specialising in quantum computing. "
                    "Answer the user's question using only the provided context. "
                    "Cite the source filename and page number for each key claim. "
                    "If the context does not contain enough information, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {clean_q}",
            },
        ],
    )

    answer = response.choices[0].message.content
    sources = sorted({doc.metadata.get("source", "?") for doc in top_docs})

    print(f"\nAnswer:\n{answer}")
    print(f"\n[sources: {', '.join(sources)}{filter_note}]")

    # Print reranking debug table
    print(f"\n{'─'*64}")
    print(f"  Reranking scores  (fetched {len(candidates)} → kept top {RERANK_TOP_N})")
    print(f"{'─'*64}")
    for rank, (score, doc) in enumerate(scored[:RERANK_TOP_N], 1):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"  #{rank}  score={score:+.4f}  [{src}, p.{page}]")
        print(f"       \"{snippet}...\"")
    print(f"{'─'*64}\n")


def main():
    if not os.path.exists(PAPERS_DIR):
        print(f"ERROR: {PAPERS_DIR}/ folder not found.")
        return

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = build_or_load_vectorstore(embeddings)

    print(f"Loading reranker: {RERANKER_MODEL} ...")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("  Ready.\n")

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    print("ResearchMind + Reranker — type 'exit' or 'quit' to stop.")
    print(f"Retrieval: top-{RETRIEVAL_K} from ChromaDB → rerank → top-{RERANK_TOP_N} to LLM")
    print("Tip: prefix with 'what does <file>.pdf say about' to filter by paper.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer_question(client, vectorstore, reranker, question)


if __name__ == "__main__":
    main()
