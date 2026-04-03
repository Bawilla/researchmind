"""
ResearchMind — Day 6: RAGAS Evaluation Pipeline

Runs 10 QEC questions through the full RAG pipeline (retrieve → rerank → generate),
evaluates each answer with RAGAS (faithfulness, answer_relevancy, context_recall),
prints a score table, and saves results to evaluation_results.json.
"""

import os
import json
import math
import warnings
import textwrap
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from groq import Groq

PAPERS_DIR = "./papers"
CHROMA_DIR = "./chroma_db6"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
RETRIEVAL_K = 10
RERANK_TOP_N = 3
RESULTS_FILE = "evaluation_results.json"

SYSTEM_PROMPT = (
    "You are a helpful research assistant specialising in quantum computing and quantum error correction. "
    "Answer the user's question using only the provided context from scientific papers. "
    "Be concise and cite source filenames and page numbers for key claims. "
    "If the context does not contain enough information, say so."
)

# ---------------------------------------------------------------------------
# 10 evaluation Q&A pairs with ground-truth reference answers
# ---------------------------------------------------------------------------
EVAL_QA = [
    {
        "question": "What is quantum error correction?",
        "ground_truth": (
            "Quantum error correction protects quantum information from errors by encoding it "
            "redundantly across multiple physical qubits. Errors are detected by measuring "
            "stabilizer operators without disturbing the encoded information, and then corrected "
            "based on the resulting syndrome."
        ),
    },
    {
        "question": "What is a stabilizer code?",
        "ground_truth": (
            "A stabilizer code is a quantum error-correcting code defined by a set of commuting "
            "Pauli operators called stabilizers. The code space consists of states that are +1 "
            "eigenstates of all stabilizer generators. Errors are detected by measuring the "
            "stabilizers to obtain a syndrome and corrected accordingly."
        ),
    },
    {
        "question": "How does the surface code correct errors?",
        "ground_truth": (
            "The surface code encodes logical qubits in a 2D lattice using nearest-neighbor "
            "X-type and Z-type stabilizer measurements to detect bit-flip and phase-flip errors. "
            "Correction is performed classically using minimum weight perfect matching on "
            "syndrome measurement outcomes. Errors only need to be corrected when they affect "
            "measurement outcomes."
        ),
    },
    {
        "question": "What is the threshold error rate for fault-tolerant quantum computing?",
        "ground_truth": (
            "The fault-tolerance threshold is the physical error rate below which quantum error "
            "correction can suppress logical errors arbitrarily. For the surface code this "
            "threshold is approximately 1%, meaning physical gate error rates must be below "
            "roughly 1% for error correction to be beneficial."
        ),
    },
    {
        "question": "What is the distance of a quantum error-correcting code?",
        "ground_truth": (
            "The distance d of a quantum error-correcting code is the minimum weight of a "
            "logical operator not detectable by the stabilizers. A distance-d code can detect "
            "errors on up to d-1 qubits and correct errors on up to floor((d-1)/2) qubits."
        ),
    },
    {
        "question": "What is the toric code?",
        "ground_truth": (
            "The toric code is a topological quantum error-correcting code defined on a 2D "
            "torus where qubits sit on edges of a lattice. Stabilizers are vertex operators "
            "(X-type) and plaquette operators (Z-type). It can correct any error pattern "
            "lighter than half the system size along any non-trivial cycle."
        ),
    },
    {
        "question": "What are the main types of quantum errors?",
        "ground_truth": (
            "The main quantum error types are bit-flip (X) errors, phase-flip (Z) errors, and "
            "combined bit-and-phase-flip (Y) errors. Any single-qubit error can be decomposed "
            "into Pauli X, Y, Z components, so correcting these three types is sufficient for "
            "general single-qubit error correction."
        ),
    },
    {
        "question": "What is fault-tolerant quantum computing?",
        "ground_truth": (
            "Fault-tolerant quantum computing performs reliable computations despite physical "
            "qubit and gate errors. It requires that errors introduced at each computational "
            "step do not propagate catastrophically, achieved via quantum error-correcting codes "
            "and fault-tolerant gate implementations such that errors remain correctable."
        ),
    },
    {
        "question": "How does minimum weight perfect matching work in quantum error correction?",
        "ground_truth": (
            "Minimum weight perfect matching (MWPM) is a classical decoding algorithm for "
            "surface codes. It takes syndrome measurement outcomes—locations where stabilizer "
            "measurements flip sign—and finds the lowest-weight pairing of defects. The "
            "corresponding error pattern is then applied as a correction."
        ),
    },
    {
        "question": "How does magic state distillation enable universal quantum computation?",
        "ground_truth": (
            "Magic state distillation produces high-fidelity non-Clifford resource states "
            "(magic states) by consuming many noisy copies and distilling them into fewer "
            "higher-quality ones. These magic states are then injected into fault-tolerant "
            "circuits via gate teleportation to implement non-Clifford gates like the T gate, "
            "enabling universal quantum computation on top of the Clifford group."
        ),
    },
]


# ---------------------------------------------------------------------------
# Indexing (identical pattern to previous mains)
# ---------------------------------------------------------------------------
def infer_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def build_or_load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing ChromaDB from {CHROMA_DIR}...")
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        print(f"  Loaded {vs._collection.count()} chunks\n")
        return vs

    print(f"Indexing all PDFs in {PAPERS_DIR}/...")
    pdf_files = sorted(Path(PAPERS_DIR).glob("*.pdf"))
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

    print(f"\nEmbedding {len(all_chunks)} total chunks into {CHROMA_DIR}/ ...")
    vs = Chroma.from_documents(
        documents=all_chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )
    print(f"  Done — {vs._collection.count()} chunks stored.\n")
    return vs


# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------
def retrieve_and_rerank(question: str, vs: Chroma, reranker: CrossEncoder):
    candidates = vs.similarity_search(question, k=RETRIEVAL_K)
    if not candidates:
        return [], []
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:RERANK_TOP_N]]
    return top_docs, scored


# ---------------------------------------------------------------------------
# RAG pipeline: question → (answer, contexts)
# ---------------------------------------------------------------------------
def run_rag(question: str, vs: Chroma, reranker: CrossEncoder, groq_client: Groq):
    top_docs, _ = retrieve_and_rerank(question, vs, reranker)
    if not top_docs:
        return "", []

    context_parts = []
    context_strings = []
    for doc in top_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{src}, p.{page}]\n{doc.page_content}")
        context_strings.append(doc.page_content)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{chr(10).join(context_parts)}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content
    return answer, context_strings


# ---------------------------------------------------------------------------
# RAGAS evaluation (supports both 0.1.x and 0.2.x APIs)
# ---------------------------------------------------------------------------
def run_ragas_evaluation(records: list) -> dict:
    """
    records: list of dicts with keys question, answer, contexts, ground_truth
    Returns dict of metric_name → list of per-question scores
    """
    from langchain_groq import ChatGroq

    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Try RAGAS 0.2.x API first
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm = LangchainLLMWrapper(groq_llm)
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            for r in records
        ]
        dataset = EvaluationDataset(samples=samples)

        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            LLMContextRecall(llm=ragas_llm),
        ]

        print("Running RAGAS evaluation (0.2.x API)...")
        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        return {
            "faithfulness":      df["faithfulness"].tolist(),
            "answer_relevancy":  df["answer_relevancy"].tolist(),
            "context_recall":    df["context_recall"].tolist(),
        }

    except (ImportError, AttributeError):
        # Fall back to RAGAS 0.1.x API
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas.llms import LangchainLLMWrapper
        from datasets import Dataset

        ragas_llm = LangchainLLMWrapper(groq_llm)
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        context_recall.llm = ragas_llm

        hf_dataset = Dataset.from_dict(
            {
                "question":     [r["question"] for r in records],
                "answer":       [r["answer"] for r in records],
                "contexts":     [r["contexts"] for r in records],
                "ground_truth": [r["ground_truth"] for r in records],
            }
        )

        print("Running RAGAS evaluation (0.1.x API)...")
        result = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        df = result.to_pandas()

        return {
            "faithfulness":     df["faithfulness"].tolist(),
            "answer_relevancy": df["answer_relevancy"].tolist(),
            "context_recall":   df["context_recall"].tolist(),
        }


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------
def print_results_table(records: list, scores: dict) -> None:
    W = 46   # question column width
    print("\n" + "═" * 82)
    print(f"  {'QUESTION':<{W}}  {'FAITH':>6}  {'RELEVANCY':>9}  {'RECALL':>7}")
    print("═" * 82)

    for i, rec in enumerate(records):
        q = textwrap.shorten(rec["question"], width=W, placeholder="…")
        fa = scores["faithfulness"][i]
        re = scores["answer_relevancy"][i]
        rc = scores["context_recall"][i]
        fa_s = f"{fa:.3f}" if fa is not None else "  N/A"
        re_s = f"{re:.3f}" if re is not None else "  N/A"
        rc_s = f"{rc:.3f}" if rc is not None else "  N/A"
        print(f"  {q:<{W}}  {fa_s:>6}  {re_s:>9}  {rc_s:>7}")

    print("─" * 82)

    def avg(lst):
        vals = [v for v in lst if v is not None and not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    fa_avg = avg(scores["faithfulness"])
    re_avg = avg(scores["answer_relevancy"])
    rc_avg = avg(scores["context_recall"])
    print(f"  {'AVERAGE':<{W}}  {fa_avg:>6.3f}  {re_avg:>9.3f}  {rc_avg:>7.3f}")
    print("═" * 82)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(PAPERS_DIR):
        print(f"ERROR: {PAPERS_DIR}/ not found.")
        return

    # 1. Load models
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = build_or_load_vectorstore(embeddings)

    print(f"Loading reranker: {RERANKER_MODEL}...")
    reranker = CrossEncoder(RERANKER_MODEL)
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("  Ready.\n")

    # 2. Run RAG pipeline for all 10 questions
    print(f"Running {len(EVAL_QA)} questions through the RAG pipeline...\n")
    records = []
    for i, item in enumerate(EVAL_QA, 1):
        q = item["question"]
        print(f"  [{i:02d}/{len(EVAL_QA)}] {q}")
        answer, contexts = run_rag(q, vs, reranker, groq_client)
        records.append(
            {
                "question":     q,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": item["ground_truth"],
            }
        )

    print(f"\nAll {len(records)} answers generated.\n")

    # 3. RAGAS evaluation
    scores = run_ragas_evaluation(records)

    # 4. Print table
    print_results_table(records, scores)

    # 5. Compute averages
    def avg(lst):
        vals = [v for v in lst if v is not None and not math.isnan(v)]
        return round(sum(vals) / len(vals), 4) if vals else None

    averages = {
        "faithfulness":     avg(scores["faithfulness"]),
        "answer_relevancy": avg(scores["answer_relevancy"]),
        "context_recall":   avg(scores["context_recall"]),
    }

    print("Average scores:")
    for metric, val in averages.items():
        print(f"  {metric:<20} {val:.4f}")
    print()

    # 6. Save to JSON
    output = {
        "model":   GROQ_MODEL,
        "papers":  sorted(p.name for p in Path(PAPERS_DIR).glob("*.pdf")),
        "metrics": averages,
        "results": [
            {
                "question":     r["question"],
                "ground_truth": r["ground_truth"],
                "answer":       r["answer"],
                "contexts":     r["contexts"],
                "scores": {
                    "faithfulness":     scores["faithfulness"][i],
                    "answer_relevancy": scores["answer_relevancy"][i],
                    "context_recall":   scores["context_recall"][i],
                },
            }
            for i, r in enumerate(records)
        ],
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Full results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
