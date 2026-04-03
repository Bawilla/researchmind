import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

PDF_PATH = "paper.pdf"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5


def load_and_index(pdf_path: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing ChromaDB from {CHROMA_DIR}...")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        print(f"  Loaded {vectorstore._collection.count()} chunks\n")
        return vectorstore

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    print("Embedding chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Stored in {CHROMA_DIR}\n")
    return vectorstore


def answer_question(client: Groq, vectorstore: Chroma, question: str) -> None:
    docs = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Answer the user's question using only the provided context. "
                    "If the context does not contain enough information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer:\n{answer}")
    print(f"\n[Retrieved {len(docs)} chunk(s) for this question]\n")


def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: {PDF_PATH} not found. Please place a PDF at {PDF_PATH}.")
        return

    vectorstore = load_and_index(PDF_PATH)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    print("ResearchMind QA — type 'exit' or 'quit' to stop.\n")
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
