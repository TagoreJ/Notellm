import streamlit as st
import google.generativeai as genai
import nbformat
from PyPDF2 import PdfReader
import os
import tempfile
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")

# API Key validation
if not API_KEY:
    st.error("❌ **GEMINI_API_KEY not found!**")
    st.markdown("""
    ### Create `.env` file in the same directory:
    ```
    GEMINI_API_KEY=your_actual_api_key_here
    ```
    Get your FREE key: [Google AI Studio](https://aistudio.google.com/app/apikey)
    """)
    st.stop()

# Initialize Gemini with updated model for 2025 - free tier compatible
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    st.sidebar.success("✅ Gemini 2.0 Connected!")
except Exception as e:
    st.error(f"❌ Gemini API Error: {str(e)}")
    st.info("💡 Try updating your SDK: pip install --upgrade google-generativeai")
    st.stop()

def extract_notebook_content(nb):
    """Extract content from Jupyter notebook"""
    content = "# 📓 JUPYTER NOTEBOOK\n\n"
    try:
        for i, cell in enumerate(nb.cells, 1):
            if cell.cell_type == 'markdown':
                content += f"## Markdown Cell {i}\n{cell.source}\n\n"
            elif cell.cell_type == 'code':
                content += f"## Code Cell {i}\n``````\n\n"
                if cell.get('outputs'):
                    outputs = json.dumps(cell.outputs, indent=2)[:1000]
                    content += f"### Outputs:\n{outputs}\n\n"
        return content
    except Exception as e:
        return f"Error parsing notebook: {str(e)}"

def extract_text_content(file_path, file_name):
    """Extract text from text files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"# 📄 {Path(file_name).stem.upper()}\n\n{content}"
    except:
        return "Error reading text file"

def extract_pdf_content(file_path):
    """Extract content from PDF"""
    try:
        reader = PdfReader(file_path)
        content = "# 📄 PDF DOCUMENT\n\n"
        for i, page in enumerate(reader.pages[:10], 1):  # Limit pages
            try:
                text = page.extract_text()
                if text and text.strip():
                    content += f"## Page {i}\n{text[:2000]}\n\n"
            except:
                continue
        return content if content != "# 📄 PDF DOCUMENT\n\n" else "No text extracted from PDF"
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract content"""
    file_name = uploaded_file.name.lower()
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    try:
        if file_name.endswith('.ipynb'):
            nb = nbformat.read(tmp_path, as_version=4)
            return extract_notebook_content(nb)
        elif file_name.endswith(('.txt', '.py', '.md', '.r', '.js', '.cpp', '.java', '.json')):
            return extract_text_content(tmp_path, file_name)
        elif file_name.endswith('.pdf'):
            return extract_pdf_content(tmp_path)
        else:
            return f"Unsupported file: {file_name}"
    finally:
        os.unlink(tmp_path)

def generate_ai_response(prompt, context):
    """Generate response using Gemini with file context"""
    try:
        system_prompt = f"""
You are an expert document analysis assistant. Answer questions based ONLY on the provided context.
Be specific, reference code sections, explain concepts clearly, and provide helpful insights.

CONTEXT FROM UPLOADED FILE:
{context}

USER QUESTION: {prompt}

Respond helpfully and accurately using only the context above.
"""
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.set_page_config(
    page_title="Notebook LLM",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📓 Notebook LLM - AI Document Assistant")
st.markdown("Upload files and chat about their content using Google Gemini 2.0!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Supported Formats")
    st.markdown("""
    - 📓 Jupyter `.ipynb`
    - 💻 Code `.py`, `.js`, `.cpp`, `.java`, `.R`, `.json`
    - 📝 Text `.txt`, `.md`
    - 📄 PDF `.pdf`
    """)
    st.markdown("---")
    st.info("🆓 Free Tier: 15-60 RPM, 1M tokens")
    if st.button("🔑 Get API Key"):
        st.markdown("[Google AI Studio](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    if st.button("ℹ️ Tips"):
        st.markdown("""
        **Great prompts:**
        - "Summarize this notebook"
        - "Explain the main algorithm"
        - "What libraries are used?"
        - "Debug this code"
        - "Generate similar examples"
        """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Document",
        type=['ipynb', 'py', 'txt', 'md', 'r', 'js', 'cpp', 'java', 'json', 'pdf'],
        help="Choose a file to analyze"
    )

with col2:
    st.info("💡 Example Questions:")
    examples = [
        "Summarize this document",
        "Explain the main function",
        "What does this code do?",
        "Find bugs or issues"
    ]
    for example in examples:
        st.write(f"• {example}")

# Session state initialization
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = ""

# File processing
if uploaded_file is not None:
    with st.spinner("🔄 Processing file..."):
        content = process_uploaded_file(uploaded_file)
        st.session_state.file_content = content
        st.session_state.filename = uploaded_file.name

    if not content.startswith(("Error", "Unsupported")):
        st.success(f"✅ Loaded: **{st.session_state.filename}**")

        # File preview
        with st.expander("📋 Document Preview", expanded=False):
            st.text_area(
                "Content",
                st.session_state.file_content[:4000],
                height=300,
                disabled=True
            )

        # Chat interface
        st.subheader("💬 Ask Questions About Your File")

        # Chat history
        for msg in st.session_state.messages[-8:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing..."):
                    response = generate_ai_response(prompt, st.session_state.file_content)
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

        # Controls
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.experimental_rerun()
        with col_btn2:
            if st.button("🔄 Reprocess File"):
                st.session_state.file_content = ""
                st.experimental_rerun()
    else:
        st.error(f"❌ {content}")

# Welcome message
if not st.session_state.file_content or uploaded_file is None:
    st.info("""
    🚀 Get Started:
    1. Upload a Jupyter notebook, code file, PDF, or text document
    2. Wait for content extraction
    3. Ask questions in the chat below
    4. Get AI-powered analysis!
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    📓 Notebook LLM | Powered by Streamlit + Google Gemini 2.0
    Updated for October 2025 | Free tier compatible 
    """, unsafe_allow_html=True)
