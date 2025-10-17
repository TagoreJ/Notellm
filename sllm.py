import streamlit as st
import tempfile
import time
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load LLMs and models once
@st.cache_resource(show_spinner=False)
def load_distilgpt2():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

@st.cache_resource(show_spinner=False)
def load_bart_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def create_chroma_collection():
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".chromadb"
    ))
    return client.get_or_create_collection(name="pdf_chunks", embedding_function=None)


def filter_redundant_sentences(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    seen = set()
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            filtered.append(s)
            seen.add(s)
    return ' '.join(filtered)


def summarize_text(text, summarizer):
    progress = st.progress(0, text="Summarizing with BART...")
    for i in range(5):
        time.sleep(0.2)
        progress.progress((i + 1) * 20)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    progress.progress(100, text="Summary complete")
    return summary


st.title("📚 Local PDF Q&A with LangChain + DistilGPT2 + Chroma")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:

    # Save uploaded PDF to temp file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    embeddings = load_embeddings()

    collection = create_chroma_collection()
    # Clear any previous records before adding new docs
    collection.delete(where={})
    collection.add(documents=texts, embeddings=[embeddings.embed(t) for t in texts])
    collection.persist()

    distilgpt2 = load_distilgpt2()
    bart_summarizer = load_bart_summarizer()

    local_llm = HuggingFacePipeline(pipeline=distilgpt2)
    qa = RetrievalQA.from_chain_type(llm=local_llm, retriever=collection.as_retriever())

    query = st.text_input("Ask a question about your PDF:")

    if query:
        start = time.time()
        raw_answer = qa.run(query)
        elapsed = time.time() - start

        option = st.radio("Refine answer with BART summarizer?", ("No", "Yes"))
        final_answer = raw_answer

        if option == "Yes":
            final_answer = summarize_text(raw_answer, bart_summarizer)

        final_answer = filter_redundant_sentences(final_answer)

        st.markdown("### Answer:")
        st.write(final_answer)
        st.caption(f"⏳ Generated in {elapsed:.2f} seconds")

else:
    st.info("Upload a PDF to start question answering.")

st.markdown("---")
st.caption("Local LangChain PDF QA with DistilGPT2 + Chroma | October 2025")
