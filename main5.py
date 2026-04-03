"""
ResearchMind — Day 5: Gradio UI with file upload, reranking, and conversation memory.

Architecture:
  - Gradio and all non-ML imports happen at module load time.
  - ML models (embeddings, cross-encoder) are loaded lazily on first PDF upload
    so Gradio's startup (pydantic-core, starlette, etc.) settles its memory
    footprint before the heavy models are mapped into memory.
"""

import os
import shutil
from pathlib import Path

import gradio as gr

CHROMA_DIR = "./chroma_db5"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
RETRIEVAL_K = 10
RERANK_TOP_N = 3
MEMORY_TURNS = 4

SYSTEM_PROMPT = (
    "You are a helpful research assistant specialising in scientific literature. "
    "Answer the user's question using only the provided context from the uploaded paper(s). "
    "Cite the source filename and page number for every key claim. "
    "Use the conversation history to resolve follow-up questions and pronouns. "
    "If the context does not contain enough information, say so clearly."
)

# Lazy singletons — populated on first use
_embeddings = None
_reranker = None
_groq = None


def _get_models():
    """Load ML models once; subsequent calls return the cached instances."""
    global _embeddings, _reranker, _groq

    if _embeddings is None:
        import warnings
        warnings.filterwarnings("ignore")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from sentence_transformers import CrossEncoder
        from groq import Groq

        print("Loading embedding model …")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"Loading reranker: {RERANKER_MODEL} …")
        _reranker = CrossEncoder(RERANKER_MODEL)
        _groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print("All models ready.")

    return _embeddings, _reranker, _groq


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_pdf(file_path: str):
    """
    Called when the user uploads a PDF.
    Returns (vectorstore, status_md, cleared_chat, cleared_history, cleared_sources).
    """
    if file_path is None:
        return None, "*No file selected.*", [], [], ""

    import warnings
    warnings.filterwarnings("ignore")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    embeddings, _, _ = _get_models()

    # Fresh DB per upload
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    original_name = Path(file_path).name

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = original_name
        page.metadata["title"] = (
            Path(original_name).stem.replace("_", " ").replace("-", " ").title()
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    status = (
        f"**Indexed:** `{original_name}`  \n"
        f"{len(pages)} pages → **{len(chunks)} chunks** stored in `{CHROMA_DIR}/`  \n"
        f"Ready to answer questions."
    )
    return vs, status, [], [], ""


# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------
def _retrieve_and_rerank(question: str, vs):
    _, reranker, _ = _get_models()
    candidates = vs.similarity_search(question, k=RETRIEVAL_K)
    if not candidates:
        return [], []
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:RERANK_TOP_N]]
    return top_docs, scored


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------
def chat(question: str, chat_history: list, history: list, vs):
    question = question.strip()
    if not question:
        return chat_history, history, "", ""

    if vs is None:
        reply = "Please upload a PDF first using the panel on the left."
        return chat_history + [[question, reply]], history, "", ""

    _, _, groq_client = _get_models()

    top_docs, scored = _retrieve_and_rerank(question, vs)

    if not top_docs:
        reply = "No relevant content found. Try rephrasing your question."
        return chat_history + [[question, reply]], history, "", ""

    # Build context with citations
    context_parts = []
    for doc in top_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{src}, p.{page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # Build prompt: system + last N exchanges + current question (with context)
    recent = history[-(MEMORY_TURNS * 2):]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *recent,
        {
            "role": "user",
            "content": f"Retrieved context:\n{context}\n\nQuestion: {question}",
        },
    ]

    response = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages)
    answer = response.choices[0].message.content

    # Update plain history (no context blobs — stays concise across turns)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    exchanges = len(history) // 2

    # Sources + reranking panel
    sources_lines = [
        f"### Sources & Reranking  "
        f"*(exchange {exchanges} | fetched {min(RETRIEVAL_K, len(scored))}"
        f" → kept top {RERANK_TOP_N})*\n"
    ]
    for rank, (score, doc) in enumerate(scored[:RERANK_TOP_N], 1):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:150].replace("\n", " ")
        sources_lines.append(
            f"**#{rank}** &nbsp; `score={score:+.4f}` &nbsp; `{src}` p.{page}"
        )
        sources_lines.append(f"> {snippet}…\n")
    sources_md = "\n".join(sources_lines)

    return chat_history + [[question, answer]], history, sources_md, ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="ResearchMind", theme=gr.themes.Soft()) as demo:

        vs_state = gr.State(None)
        history_state = gr.State([])

        gr.Markdown(
            "# ResearchMind\n"
            "**Scientific Paper RAG** — upload any PDF and have a conversation about it.\n\n"
            "_Pipeline: bi-encoder top-10 → cross-encoder rerank → top-3 to Groq LLM_"
        )

        with gr.Row():
            # ── left: upload ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Upload Paper")
                file_upload = gr.File(
                    label="Drop a PDF here",
                    file_types=[".pdf"],
                    type="filepath",
                )
                index_status = gr.Markdown("*No paper indexed yet.*")

            # ── right: chat ───────────────────────────────────────────────
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=420,
                    show_copy_button=True,
                )
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="Ask a question about the paper…",
                        label="",
                        scale=5,
                        autofocus=True,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                clear_btn = gr.Button("Clear conversation", size="sm")

        # ── sources panel (full width) ────────────────────────────────────
        gr.Markdown("---")
        sources_panel = gr.Markdown("*Sources and reranking scores will appear here.*")

        # ── events ───────────────────────────────────────────────────────
        file_upload.change(
            fn=index_pdf,
            inputs=[file_upload],
            outputs=[vs_state, index_status, chatbot, history_state, sources_panel],
        )

        submit_btn.click(
            fn=chat,
            inputs=[question_box, chatbot, history_state, vs_state],
            outputs=[chatbot, history_state, sources_panel, question_box],
        )

        question_box.submit(
            fn=chat,
            inputs=[question_box, chatbot, history_state, vs_state],
            outputs=[chatbot, history_state, sources_panel, question_box],
        )

        clear_btn.click(
            fn=lambda: ([], [], "*Sources and reranking scores will appear here.*"),
            inputs=[],
            outputs=[chatbot, history_state, sources_panel],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    print("Starting ResearchMind UI at http://127.0.0.1:7860")
    app.launch(share=False, inbrowser=True)
