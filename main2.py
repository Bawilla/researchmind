import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

PAPERS_DIR = "./papers"
CHROMA_DIR = "./chroma_db2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5


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
            # page number is already set by PyPDFLoader as 'page' (0-indexed); keep it

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"{len(pages)} pages → {len(chunks)} chunks")

    return all_chunks


def build_or_load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing ChromaDB from {CHROMA_DIR}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
        print(f"  Loaded {vectorstore._collection.count()} chunks\n")
        return vectorstore

    print(f"Indexing all PDFs in {PAPERS_DIR}/...")
    chunks = load_all_papers(PAPERS_DIR)
    print(f"\nEmbedding {len(chunks)} total chunks into {CHROMA_DIR}/ ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    total = vectorstore._collection.count()
    print(f"  Done — {total} chunks stored.\n")
    return vectorstore


# ---------------------------------------------------------------------------
# Metadata filter parsing
# Supports:  "what does surface_codes.pdf say about X"
#            "according to stabilizer_codes.pdf, what is Y"
# ---------------------------------------------------------------------------
_FILTER_PATTERNS = [
    r"what does ([a-zA-Z0-9_\-]+\.pdf) say about (.+)",
    r"according to ([a-zA-Z0-9_\-]+\.pdf)[,\s]+(.+)",
    r"in ([a-zA-Z0-9_\-]+\.pdf)[,\s]+(.+)",
]


def parse_filter(question: str):
    """Return (query_text, chroma_where_filter | None)."""
    for pattern in _FILTER_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            filename, topic = m.group(1), m.group(2).strip("?. ")
            return topic, {"source": filename}
    return question, None


def answer_question(client: Groq, vectorstore: Chroma, question: str) -> None:
    clean_q, where_filter = parse_filter(question)

    if where_filter:
        docs = vectorstore.similarity_search(clean_q, k=TOP_K, filter=where_filter)
        filter_note = f" | filtered to {where_filter['source']}"
    else:
        docs = vectorstore.similarity_search(clean_q, k=TOP_K)
        filter_note = ""

    if not docs:
        print("\n[No relevant chunks found — try a different question or filename.]\n")
        return

    # Build context with inline source citations
    context_parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{src}, p.{page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

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
    sources = sorted({doc.metadata.get("source", "?") for doc in docs})

    print(f"\nAnswer:\n{answer}")
    print(f"\n[{len(docs)} chunk(s){filter_note} | sources: {', '.join(sources)}]\n")


def main():
    if not os.path.exists(PAPERS_DIR):
        print(f"ERROR: {PAPERS_DIR}/ folder not found.")
        return

    vectorstore = build_or_load_vectorstore()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    print("ResearchMind Multi-Doc QA — type 'exit' or 'quit' to stop.")
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

        answer_question(client, vectorstore, question)


if __name__ == "__main__":
    main()
