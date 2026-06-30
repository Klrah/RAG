import os
import pickle
import streamlit as nn
import streamlit as st
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from guardrails import validate_input, validate_output

# Configuration Constants
GROQ_MODEL = "llama-3.1-8b-instant"  # Fastest execution and minimal token consumption
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "./indices"

st.set_page_config(page_title="Linux Copilot", page_icon="🐧", layout="wide")

# Ensure API Key exists
if "GROQ_API_KEY" not in os.environ and not st.sidebar.text_input("Groq API Key", type="password", key="sidebar_api_key"):
    st.warning("Please configure your GROQ_API_KEY environment variable or enter it in the sidebar to proceed.")
    st.stop()

api_key = os.environ.get("GROQ_API_KEY") or st.session_state.sidebar_api_key

# Singleton Resource Loader for Production Optimization
@st.cache_resource(show_spinner="Initializing retrieval engines...")
def initialize_rag_system():
    if not (os.path.exists(f"{INDEX_DIR}/faiss.index") and 
            os.path.exists(f"{INDEX_DIR}/bm25.pkl") and 
            os.path.exists(f"{INDEX_DIR}/chunks.pkl")):
        return None

    groq_client = Groq(api_key=api_key)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    faiss_index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    
    with open(f"{INDEX_DIR}/bm25.pkl", "rb") as f:
        bm25_index = pickle.load(f)
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
        
    return {
        "client": groq_client,
        "embedder": embedder,
        "faiss": faiss_index,
        "bm25": bm25_index,
        "chunks": chunks
    }

rag = initialize_rag_system()

if rag == None:
    st.error("Retrieval indexes missing. Please run `python ingest.py` first to generate data points.")
    st.stop()

# Helper Fusion & Generation Logic
def reciprocal_rank_fusion(dense_ranks, sparse_ranks, total_docs, k=60):
    fused_scores = {i: 0 for i in range(total_docs)}
    for rank, doc_id in enumerate(dense_ranks):
        fused_scores[doc_id] += 1 / (k + rank)
    for rank, doc_id in enumerate(sparse_ranks):
        fused_scores[doc_id] += 1 / (k + rank)
    return sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

def run_hybrid_retrieval(query: str, top_k: int = 3) -> str:
    # 1. Dense retrieval
    query_emb = rag["embedder"].encode([query])
    _, dense_indices = rag["faiss"].search(query_emb, top_k * 2)
    
    # 2. Sparse retrieval
    tokenized_query = query.lower().split()
    bm25_scores = rag["bm25"].get_scores(tokenized_query)
    sparse_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]

    # 3. Reciprocal Rank Fusion
    fused_indices = reciprocal_rank_fusion(dense_indices[0], sparse_indices, len(rag["chunks"]))
    retrieved_chunks = [rag["chunks"][i] for i in fused_indices[:top_k]]
    
    return "\n\n".join(retrieved_chunks)

# UI Elements Layout
st.title("🐧 Production Hybrid-RAG Linux Copilot")
st.caption("Context-grounded technical response system using FAISS + BM25 RRF Pipeline.")

# Sidebar status panels
st.sidebar.title("System Parameters")
st.sidebar.info(f"**LLM Backend:** {GROQ_MODEL}\n\n**Embedding Architecture:** {EMBEDDING_MODEL}\n\n**Data Source:** The Linux Command Line Book")

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Maintain continuous session states
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display entire log sequence smoothly
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "warning" in msg:
            st.warning(msg["warning"])

# Interaction Flow
if user_input := st.chat_input("Ask a Linux terminal or architecture question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Execute Pipeline
    try:
        validate_input(user_input)
        
        with st.spinner("Retrieving domain context & compiling answer..."):
            context_payload = run_hybrid_retrieval(user_input)
            
            # Optimized targeted prompting to conserve input token limits
            system_instruction = """You are a precise Linux operating system CLI assistant. 
            Answer technical queries using ONLY the provided Context. 

            CRITICAL RULES:
            1. Ignore typos in the user's query and figure out their intent.
            2. The context may contain Table of Contents entries (e.g., text ending in dotted lines and page numbers). DO NOT just output a Table of Contents line.
            3. If the context gives you a topic (like 'Copying files') but doesn't show the actual command (like 'cp'), use your internal knowledge to provide the command, but state clearly that you inferred it because the PDF context only provided a chapter title."""
            user_payload = f"Context:\n{context_payload}\n\nQuery: {user_input}"
            
            completion = rag["client"].chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_payload}
                ],
                max_tokens=350,
                temperature=0.0
            )
            
            raw_response = completion.choices[0].message.content
            final_response, guardrail_triggered = validate_output(raw_response)
            
        with st.chat_message("assistant"):
            st.markdown(final_response)
            msg_entry = {"role": "assistant", "content": final_response}
            
            if guardrail_triggered:
                warning_text = "⚠️ **GUARDRAIL WARNING**: Destructive command syntax patterns discovered in output text execution. Validate safely inside an isolated testing environment before performing."
                st.warning(warning_text)
                msg_entry["warning"] = warning_text
                
            st.session_state.messages.append(msg_entry)

    except ValueError as e:
        st.error(f"Execution Halted: {str(e)}")
    except Exception as e:
        st.error("An internal processing mismatch occurred. Check execution trace logs.")