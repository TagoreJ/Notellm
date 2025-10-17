import streamlit as st
import tempfile
import time
import re
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === Model loading with caching ===
@st.cache_resource(show_spinner=False)
def load_distilgpt2():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

@st.cache_resource(show_spinner=False)
def load_bart_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def create_chroma_collection():
    import chromadb
    client = chromadb.Client()
    collection = client.get_or_create_collection("pdf_chunks")
    return collection

def filter_redundant_sentences(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    seen = set()
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            filtered.append(s)
            seen.add(s)
    return " ".join(filtered)

def summarize_text(text, summarizer):
    progress = st.progress(0, text="Summarizing with BART...")
    for i in range(5):
        time.sleep(0.2)
        progress.progress((i + 1) * 20)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
    progress.progress(100, text="Summary complete")
    return summary

st.title("📄 Local PDF QA App with LangChain + Chroma (Streamlit Cloud Ready)")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf is not None:
    # Save to temp file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = load_embeddings()
    chunk_embeddings = [embeddings.embed(t.page_content) for t in texts]

    collection = create_chroma_collection()
    # Clear existing contents before adding new
    collection.delete(where={})
    collection.add(documents=[t.page_content for t in texts], embeddings=chunk_embeddings)

    distilgpt2 = load_distilgpt2()
    bart_summarizer = load_bart_summarizer()

    local_llm = HuggingFacePipeline(pipeline=distilgpt2)
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=collection.as_retriever())

    query = st.text_input("Ask a question about the PDF:")

    if query:
        start_time = time.time()
        with st.spinner("Generating answer..."):
            raw_answer = qa_chain.run(query)
        elapsed = time.time() - start_time

        refine = st.radio("Refine answer with BART summarizer?", ("No", "Yes"))
        final_answer = raw_answer

        if refine == "Yes":
            final_answer = summarize_text(raw_answer, bart_summarizer)

        final_answer = filter_redundant_sentences(final_answer)

        st.markdown("### Answer:")
        st.write(final_answer)
        st.caption(f"⏰ Answer generated in {elapsed:.2f} seconds")

else:
    st.info("Upload a PDF file to begin question answering.")

st.markdown("---")
st.caption("By [Your Name] — Local LangChain + DistilGPT2 + Chroma RAG | Optimized for Streamlit Cloud")
