import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import time
import re

# Load local models once
@st.cache_resource(show_spinner=False)
def load_distilgpt2():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

@st.cache_resource(show_spinner=False)
def load_bart_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# PDF text extraction and chunking
def extract_pdf_text_chunks(pdf_file, chunk_size=300, chunk_overlap=50):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages[:10]:
        text = page.extract_text()
        if text:
            full_text += text + " "
    sentences = re.split(r'(?<=[.?!])\s+', full_text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            overlap_text = " ".join(chunks[i-1].split()[-chunk_overlap:])
            overlapped_chunks.append(overlap_text + " " + chunk)
    return overlapped_chunks

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(query_embedding, index, k=3):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]

def filter_redundant_sentences(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    filtered = []
    seen = set()
    for sent in sentences:
        sent = sent.strip()
        if sent and sent not in seen:
            filtered.append(sent)
            seen.add(sent)
    return ' '.join(filtered)

def build_prompt(context_chunks, query):
    context_text = "\n\n".join(context_chunks)
    return (f"You are a helpful academic assistant. "
            f"Answer concisely based ONLY on the context below. Avoid repetition.\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{query}\n\nAnswer:")

def generate_distilgpt2_answer(prompt, generator):
    progress_bar = st.progress(0, text="Generating answer with DistilGPT2...")
    for i in range(5):
        time.sleep(0.3)
        progress_bar.progress((i+1)*20)
    output = generator(prompt, max_length=200, do_sample=True)[0]["generated_text"]
    progress_bar.progress(100, text="Generation complete")
    return filter_redundant_sentences(output)

def generate_bart_summary(text, summarizer):
    progress_bar = st.progress(0, text="Summarizing with BART...")
    for i in range(5):
        time.sleep(0.2)
        progress_bar.progress((i+1)*20)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    progress_bar.progress(100, text="Summary complete")
    return summary

# ===== Streamlit UI =====

st.title("📄 Local PDF QA with Mixed DistilGPT2 and BART Summarization")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

distilgpt2 = load_distilgpt2()
bart_summarizer = load_bart_summarizer()
embed_model = load_embedding_model()

if uploaded_file:
    with st.spinner("Extracting text and building vector index..."):
        chunks = extract_pdf_text_chunks(uploaded_file)
        chunk_embeddings = embed_model.encode(chunks)
        faiss_index = build_faiss_index(np.array(chunk_embeddings))
    st.success(f"Extracted and indexed {len(chunks)} text chunks")

    query = st.text_input("Ask a question about the PDF:")

    if query:
        start_time = time.time()
        query_embedding = embed_model.encode(query)
        top_indices = retrieve_chunks(query_embedding, faiss_index)
        retrieved_chunks = [chunks[i] for i in top_indices]

        st.markdown("### Retrieved Text Snippets:")
        for i, snippet in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Snippet {i}:** {snippet[:400]}...")

        prompt = build_prompt(retrieved_chunks, query)

        option = st.radio("Choose answer mode:", ("DistilGPT2 only", "DistilGPT2 + BART summary"))
        
        if option == "DistilGPT2 only":
            answer = generate_distilgpt2_answer(prompt, distilgpt2)
        else:
            raw_answer = generate_distilgpt2_answer(prompt, distilgpt2)
            answer = generate_bart_summary(raw_answer, bart_summarizer)

        st.markdown("### Answer:")
        st.write(answer)

        elapsed = time.time() - start_time
        st.caption(f"⏰ Answer generated in {elapsed:.2f} seconds")

else:
    st.info("Upload a PDF to ask questions and get mixed model answers.")

st.markdown("---")
st.caption("Local DistilGPT2 + BART Hybrid Summarization RAG Demo | October 2025")
