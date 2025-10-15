import streamlit as st
import google.generativeai as genai
import nbformat
from PyPDF2 import PdfReader
import os
import tempfile
import json
import requests
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
import io
import time
import re
from datetime import datetime
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available, using simple search")

try:
    import gtts
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("GEMINI_API_KEY required in secrets or .env")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    st.session_state.model_ready = True
except:
    st.session_state.model_ready = False
    st.error("Model connection failed")

class NotebookLM:
    def __init__(self):
        self.content = ""
        self.chunks = []
        self.metadata = {}
        self.sources = []
        self.guide = ""
        self.audio = None
        
    def safe_extract_text(self, uploaded_file):
        """Safe text extraction with progress"""
        file_name = uploaded_file.name.lower()
        content = ""
        
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            status_text = st.empty()
            status_text.text(f"Processing {file_name}...")
            
            if file_name.endswith('.ipynb'):
                content = self._extract_notebook(tmp_path)
            elif file_name.endswith('.pdf'):
                content = self._extract_pdf(tmp_path)
            else:
                content = self._extract_text(tmp_path)
                
            self.content = content[:100000]  # Limit size
            self.metadata = {"filename": uploaded_file.name, "type": file_name.split('.')[-1]}
            
            status_text.text("✓ Content extracted successfully")
            return True
            
        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def _extract_notebook(self, path):
        try:
            nb = nbformat.read(path, as_version=4)
            content = "NOTEBOOK:\n"
            for cell in nb.cells[:20]:
                if cell.cell_type == 'code':
                    content += f"CODE:\n{cell.source[:2000]}\n\n"
                else:
                    content += f"MARKDOWN:\n{cell.source[:1000]}\n\n"
            return content
        except:
            return "Notebook content unavailable"
    
    def _extract_pdf(self, path):
        try:
            reader = PdfReader(path)
            content = "PDF DOCUMENT:\n"
            for i, page in enumerate(reader.pages[:25]):
                try:
                    text = page.extract_text()
                    if text.strip():
                        content += f"PAGE {i+1}:\n{text[:1500]}\n\n"
                except:
                    continue
            return content
        except:
            return "PDF content unavailable"
    
    def _extract_text(self, path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return "Text content unavailable"
    
    def simple_search(self, query):
        """Simple keyword search when sklearn unavailable"""
        if not self.chunks:
            self.chunks = [self.content[i:i+2000] for i in range(0, len(self.content), 2000)]
        
        query_words = set(query.lower().split())
        best_match = max(self.chunks, key=lambda chunk: len(query_words.intersection(set(chunk.lower().split()))))
        return [best_match]
    
    def rag_search(self, query):
        """RAG search with fallback"""
        if SKLEARN_AVAILABLE and len(self.chunks) > 1:
            try:
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform(self.chunks)
                query_vec = vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, X).flatten()
                top_idx = np.argmax(similarities)
                return [self.chunks[top_idx]]
            except:
                pass
        
        # Fallback to simple search
        return self.simple_search(query)
    
    def generate_response(self, query):
        """Generate focused response"""
        try:
            context = '\n'.join(self.rag_search(query))
            
            prompt = f"""
            Using only this document context, answer concisely:
            
            CONTEXT: {context[:4000]}
            
            QUESTION: {query}
            
            Give direct, relevant answer only.
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Response error: {str(e)}"
    
    def generate_guide(self):
        """Generate study guide"""
        try:
            prompt = f"""
            Create concise study guide for:
            {self.content[:20000]}
            
            Include:
            - Key concepts
            - Main points
            - Important terms
            
            Keep brief and structured.
            """
            response = model.generate_content(prompt)
            self.guide = response.text
            return self.guide
        except:
            return "Guide generation failed"
    
    def generate_audio(self, text):
        """Generate audio with fallback"""
        if not GTTS_AVAILABLE:
            return None
        try:
            tts = gtts.gTTS(text[:300], lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes.getvalue()
        except:
            return None

# Custom CSS to match NotebookLM
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    
.main {
    font-family: 'Google Sans', sans-serif;
}

.stApp {
    background-color: #fafafa;
}

.header-section {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}

.upload-area {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    border: 2px dashed #e5e7eb;
    text-align: center;
    margin: 1rem 0;
}

.source-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.chat-bubble {
    background: white;
    border-radius: 18px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.notebooklm-blue {
    color: #1e40af;
}

.guide-section {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 4px solid #3b82f6;
    padding: 1.5rem;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.75rem 1.5rem;
}

.sidebar .sidebar-content {
    background: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# Initialize session
if "notebook" not in st.session_state:
    st.session_state.notebook = NotebookLM()
if "sources" not in st.session_state:
    st.session_state.sources = []
if "chat" not in st.session_state:
    st.session_state.chat = []

# Header - NotebookLM Style
st.markdown("""
<div class="header-section">
    <h1 style="margin: 0; font-weight: 700;">NotebookLM</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Your notebook. Your AI assistant.</p>
</div>
""", unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your first source", 
        type=['pdf', 'ipynb', 'txt', 'md', 'py'],
        help="Supports PDFs, notebooks, and text files"
    )
    
    if uploaded_file is not None and st.button("Add Source", key="add_source"):
        with st.spinner("Processing document..."):
            if st.session_state.notebook.safe_extract_text(uploaded_file):
                st.session_state.sources.append({
                    "name": uploaded_file.name,
                    "added": datetime.now().strftime("%H:%M")
                })
                st.success(f"✓ {uploaded_file.name} added")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sources list
    if st.session_state.sources:
        st.subheader("Sources")
        for i, source in enumerate(st.session_state.sources):
            st.markdown(f"""
            <div class="source-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{source['name']}</strong><br>
                        <small style="color: #6b7280;">Added {source['added']}</small>
                    </div>
                    <button style="background: #ef4444; color: white; border: none; border-radius: 6px; padding: 0.25rem 0.5rem; cursor: pointer;" onclick="this.parentElement.parentElement.remove()">×</button>
                </div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h3 style="color: #1e40af; margin-top: 0;">What can I help with?</h3>
        <ul style="color: #6b7280; line-height: 1.6;">
            <li>Summarize key points</li>
            <li>Explain concepts</li>
            <li>Create study guides</li>
            <li>Answer questions</li>
            <li>Generate audio overviews</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main tabs
if st.session_state.sources:
    tab1, tab2, tab3 = st.tabs(["Notebook", "Chat", "Guide"])
    
    with tab1:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.subheader("Notebook Guide")
        
        if st.button("Generate Guide", key="gen_guide"):
            with st.spinner("Creating guide..."):
                guide = st.session_state.notebook.generate_guide()
                st.markdown(f"### Key Concepts\n{guide}")
        
        # Audio generation
        if st.button("Generate Audio Overview", key="gen_audio"):
            with st.spinner("Creating audio..."):
                audio = st.session_state.notebook.generate_audio(st.session_state.notebook.guide or st.session_state.notebook.content[:500])
                if audio:
                    st.audio(audio)
                else:
                    st.warning("Audio generation unavailable")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Ask questions about your sources")
        
        # Chat interface
        for message in st.session_state.chat[-10:]:
            with st.chat_message(message["role"]):
                st.markdown(f'<div class="chat-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-bubble">{prompt}</div>', unsafe_allow_html=True)
            
            with st.chat_message("assistant"):
                with st.spinner("NotebookLM is thinking..."):
                    response = st.session_state.notebook.generate_response(prompt)
                    st.markdown(f'<div class="chat-bubble">{response}</div>', unsafe_allow_html=True)
                
                st.session_state.chat.append({"role": "assistant", "content": response})
    
    with tab3:
        st.subheader("Source Details")
        st.text_area("Content Preview", st.session_state.notebook.content[:2000], height=300, disabled=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6b7280; border-top: 1px solid #e5e7eb; margin-top: 2rem;">
    <p>Powered by Google Gemini • Your personal AI research assistant</p>
</div>
""", unsafe_allow_html=True)

# Clear session
if st.sidebar.button("Clear All Sources"):
    st.session_state.sources = []
    st.session_state.chat = []
    st.session_state.notebook = NotebookLM()
    st.rerun()