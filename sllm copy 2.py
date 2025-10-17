import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import time

# Load small local GPT2 model just once and cache it
@st.cache_resource(show_spinner=False)
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator

# Load SBERT embedding model once and cache it
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Extract text chunks from PDF pages
def extract_pdf_text_chunks(pdf_file, chunk_size=500, chunk_overlap=50):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages[:10]:  # limit pages
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    # Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Build FAISS index (simple L2) for chunks embeddings
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Retrieve top k relevant chunks for a query embedding
def retrieve_chunks(query_embedding, index, k=3):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]

# Format prompt for DistilGPT2 with retrieved context + query
def build_prompt(context_chunks, query):
    context_text = "\n".join(context_chunks)
    prompt = (f"Answer the question based ONLY on the context below.\n\n"
              f"CONTEXT:\n{context_text}\n\n"
              f"QUESTION: {query}\n\nAnswer:")
    return prompt

# Generate answer using local DistilGPT2
def generate_answer(prompt, generator):
    # Simulate progress bar
    progress_bar = st.progress(0)
    for i in range(5):
        time.sleep(0.3)
        progress_bar.progress((i+1)*20)
    outputs = generator(prompt, max_length=200, do_sample=True)
    progress_bar.progress(100)
    return outputs[0]['generated_text']

# Streamlit App starts here

st.title("📄 PDF + Local DistilGPT2 RAG Demo")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Upload PDF to query with local DistilGPT2")

if uploaded_file is not None:
    with st.spinner("Extracting text and building index..."):
        text_chunks = extract_pdf_text_chunks(uploaded_file)
        embed_model = load_embedding_model()
        chunk_embeddings = embed_model.encode(text_chunks)
        faiss_index = build_faiss_index(np.array(chunk_embeddings))

        st.success(f"Extracted {len(text_chunks)} chunks and built FAISS index")

    generator = load_local_llm()

    query = st.text_input("Ask a question about your PDF:")

    if query:
        start = time.time()
        query_emb = embed_model.encode(query)
        top_indices = retrieve_chunks(query_emb, faiss_index, k=3)
        retrieved_chunks = [text_chunks[i] for i in top_indices]

        prompt = build_prompt(retrieved_chunks, query)

        with st.spinner("Generating answer with DistilGPT2..."):
            answer = generate_answer(prompt, generator)

        elapsed = time.time() - start

        st.markdown("**Answer:**")
        st.write(answer)

        st.caption(f"⏰ Answer generated in {elapsed:.2f} seconds")
