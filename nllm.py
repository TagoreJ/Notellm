import streamlit as st
import google.generativeai as genai
import nbformat
from PyPDF2 import PdfReader
import os
import tempfile
import json
from io import BytesIO
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found! Add to `.env` file.")
    st.stop()

# Configure Gemini with updated model
try:
    genai.configure(api_key=API_KEY)
    # Use the latest stable model (Oct 2025)
    model = genai.GenerativeModel('gemini-2.5-flash')
    st.sidebar.success("✅ Gemini 2.5 Connected!")
except Exception as e:
    st.error(f"❌ API Error: {str(e)}")
    st.stop()

def extract_text_from_file(uploaded_file):
    """Extract content from various file types"""
    file_name = uploaded_file.name.lower()
    content = ""
    
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        if file_name.endswith('.ipynb'):
            nb = nbformat.read(tmp_file_path, as_version=4)
            content = "# 📓 JUPYTER NOTEBOOK\n\n"
            for i, cell in enumerate(nb.cells, 1):
                if cell.cell_type == 'markdown':
                    content += f"## Markdown Cell {i}\n{cell.source}\n\n"
                elif cell.cell_type == 'code':
                    content += f"## Code Cell {i}\n```python\n{cell.source}\n```\n"
                    if cell.get('outputs'):
                        content += f"### Outputs:\n{json.dumps(cell.outputs, indent=2)[:1000]}\n\n"
        
        elif file_name.endswith(('.txt', '.py', '.md', '.R', '.js', '.cpp', '.java')):
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                content = f"# 📄 {file_name.upper()}\n\n{f.read()}"
        
        elif file_name.endswith('.pdf'):
            reader = PdfReader(tmp_file_path)
            content = "# 📄 PDF DOCUMENT\n\n"
            for i, page in enumerate(reader.pages[:10], 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        content += f"## Page {i}\n{text[:2000]}\n\n"
                except:
                    continue
        
        else:
            return f"Unsupported file type: {file_name}"
    
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    return content[:300000]  # Limit for token constraints

def generate_response(prompt, file_content):
    """Generate AI response using file context"""
    try:
        system_prompt = f"""
You are an expert document and code analysis assistant. Answer questions based ONLY on the provided context.
Be specific, reference sections or code when possible, and provide helpful insights.

CONTEXT FROM UPLOADED FILE:
{file_content}

USER QUESTION: {prompt}

Provide a clear, accurate response using the context above.
"""
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit UI
st.set_page_config(
    page_title="Notebook LLM", 
    page_icon="📓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📓 Notebook LLM - File Analysis Assistant")
st.markdown("Upload documents and chat about their content using Gemini 2.5!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("**Supported Files:**")
    st.write("• 📓 Jupyter `.ipynb`")
    st.write("• 💻 Code: `.py`, `.js`, `.cpp`, `.java`")
    st.write("• 📝 Text: `.txt`, `.md`")
    st.write("• 📄 PDF files")
    
    st.markdown("---")
    st.info("🆓 **Free Tier Ready** - Uses Gemini 2.5 Flash")
    
    if st.button("🔑 Get API Key"):
        st.markdown("[Google AI Studio](https://aistudio.google.com/app/apikey)")

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload a file to analyze",
    type=['ipynb', 'py', 'txt', 'md', 'R', 'js', 'cpp', 'java', 'pdf']
)

# Session state
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = ""

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("🔄 Extracting content..."):
        content = extract_text_from_file(uploaded_file)
        st.session_state.file_content = content
        st.session_state.filename = uploaded_file.name
    
    if not content.startswith(("Error", "Unsupported")):
        st.success(f"✅ Loaded: **{st.session_state.filename}**")
        
        # File preview
        with st.expander("📋 File Preview", expanded=True):
            st.text_area(
                "Content", 
                st.session_state.file_content[:3000], 
                height=250, 
                disabled=True
            )
        
        # Chat interface
        st.subheader("💬 Ask Questions About Your File")
        
        # Display chat history
        for message in st.session_state.messages[-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your document (e.g., 'Summarize this', 'Explain the code')"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Gemini is analyzing..."):
                    response = generate_response(prompt, st.session_state.file_content)
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Upload New File"):
                st.session_state.file_content = ""
                st.session_state.messages = []
                st.session_state.filename = ""
                st.rerun()
    else:
        st.error(f"❌ {content}")

# Welcome message for new users
if uploaded_file is None and not st.session_state.messages:
    st.info("""
    🚀 **How to use Notebook LLM:**
    1. **Upload** a Jupyter notebook, code file, PDF, or text document
    2. **Wait** for content extraction (automatic)
    3. **Chat** - Ask questions about your file in the input box below
    4. **Get insights** - Gemini analyzes and responds based on your content!
    
    💡 **Example questions:**
    • "Summarize this notebook in bullet points"
    • "Explain what this function does"
    • "What libraries are used?"
    • "Find bugs in the code"
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📓 Notebook LLM | Powered by Streamlit + Gemini 2.5 Flash<br>
        <small>Free tier compatible | Oct 2025</small>
    </div>
    """, 
    unsafe_allow_html=True
)