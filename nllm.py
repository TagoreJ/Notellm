import streamlit as st
import google.generativeai as genai
import nbformat
from PyPDF2 import PdfReader
import os
import tempfile
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API key
if not API_KEY:
    st.error("❌ **GEMINI_API_KEY not found!**")
    st.markdown("""
    ### Create `.env` file:
    ```
    GEMINI_API_KEY=your_actual_api_key_here
    ```
    Get your FREE key: [Google AI Studio](https://aistudio.google.com/app/apikey)
    """)
    st.stop()

# Initialize Gemini (using flash‑lite for higher limits)
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    st.sidebar.success("✅ Gemini 2.0 Flash‑Lite Connected!")
except Exception as e:
    st.error(f"❌ Gemini API Error: {e}")
    st.info("💡 Try: pip install --upgrade google-generativeai")
    st.stop()

# ---------------- File Extraction ----------------
def extract_notebook_content(nb):
    """Extracts readable notebook content"""
    content = "# 📓 JUPYTER NOTEBOOK\n\n"
    try:
        for i, cell in enumerate(nb.cells, 1):
            if cell.cell_type == "markdown":
                content += f"## Markdown Cell {i}\n{cell.source}\n\n"
            elif cell.cell_type == "code":
                content += f"## Code Cell {i}\n``````\n\n"
                if cell.get("outputs"):
                    outputs = json.dumps(cell.outputs, indent=2)[:1000]
                    content += f"### Outputs:\n{outputs}\n\n"
    except Exception as e:
        content += f"\nError parsing notebook: {e}"
    return content

def extract_text_content(file_path, file_name):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f"# 📄 {Path(file_name).stem.upper()}\n\n{f.read()}"
    except:
        return "Error reading text file"

def extract_pdf_content(file_path):
    try:
        reader = PdfReader(file_path)
        content = "# 📄 PDF CONTENT\n\n"
        for i, page in enumerate(reader.pages[:10], 1):
            text = page.extract_text()
            if text:
                content += f"## Page {i}\n{text[:2000]}\n\n"
        return content
    except Exception as e:
        return f"Error parsing PDF: {e}"

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process uploaded file to extract text"""
    name = uploaded_file.name.lower()
    buf = uploaded_file.read()
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
        tmp.write(buf)
        path = tmp.name
    try:
        if name.endswith(".ipynb"):
            nb = nbformat.read(path, as_version=4)
            return extract_notebook_content(nb)
        elif name.endswith((".txt", ".py", ".md", ".r", ".js", ".cpp", ".java", ".json")):
            return extract_text_content(path, name)
        elif name.endswith(".pdf"):
            return extract_pdf_content(path)
        else:
            return "Unsupported file type"
    finally:
        os.unlink(path)

# ---------------- Gemini Response with Retry ----------------
def generate_ai_response(prompt, context, retries=3):
    """Generates response with rate‑limit (429) handling"""
    system_prompt = f"""
You are an expert assistant. Analyze and answer using ONLY this context.

CONTEXT:
{context}

QUESTION:
{prompt}
"""
    for attempt in range(retries):
        try:
            response = model.generate_content(system_prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):  # Rate limit exceeded
                wait_time = min(60, (attempt + 1) * 20)
                st.warning(f"⚠️ Rate limit hit — retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return f"Error generating response: {e}"
    return "⚠️ Still over rate limit. Please wait a minute and retry."

# ---------------- Streamlit Interface ----------------
st.set_page_config(page_title="Notebook LLM", page_icon="📓", layout="wide")

st.title("📓 Notebook LLM (Gemini 2.0 Flash‑Lite)")
st.caption("Chat about notebooks or PDFs — handles Gemini rate limits automatically.")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("""
**Supported Files:**
- 📓 `.ipynb`
- 💻 `.py`, `.js`, `.cpp`, `.java`, `.R`, `.json`
- 📝 `.txt`, `.md`
- 📄 `.pdf`
""")
    st.info("💡 Free Tier: 30 requests/minute, 1 M tokens with Flash‑Lite")

uploaded_file = st.file_uploader("📁 Upload Document", type=[
    "ipynb", "py", "txt", "md", "r", "js", "cpp", "java", "json", "pdf"
])

if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    with st.spinner("Processing file..."):
        data = process_uploaded_file(uploaded_file)
        st.session_state.file_content = data
        st.success(f"✅ File loaded: {uploaded_file.name}")

    with st.expander("📄 Document Preview"):
        st.text_area("Extracted content", data[:4000], height=300, disabled=True)

    # Chat
    st.subheader("💬 Ask Questions About the File")
    for msg in st.session_state.messages[-8:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                result = generate_ai_response(prompt, st.session_state.file_content)
                st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

    cols = st.columns(2)
    with cols[0]:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with cols[1]:
        if st.button("🔄 Reprocess File"):
            st.session_state.file_content = ""
            st.rerun()

else:
    st.info("📂 Upload a notebook, script, or PDF to begin.")

st.markdown("---")
st.caption("Updated October 2025 • Handles 429 errors • Gemini 2.0 Flash‑Lite ready.")
