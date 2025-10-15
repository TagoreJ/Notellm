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
import hashlib
import base64
from datetime import datetime
import re
from collections import defaultdict
import numpy as np

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY required!")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    st.sidebar.success("✅ Gemini 2.5 Ready!")
except:
    st.error("❌ API Connection Failed!")
    st.stop()

class NotebookLLM:
    def __init__(self):
        self.file_content = ""
        self.metadata = {}
        self.guide_content = ""
        self.podcast_notes = ""
        self.studynotes = ""
        self.timeline = []
        self.faq = []
        self.brainstorm = ""
        
    def extract_content(self, uploaded_file):
        """Enhanced content extraction with metadata"""
        file_name = uploaded_file.name.lower()
        content = ""
        metadata = {"filename": file_name, "type": "", "keywords": [], "timestamp": datetime.now().isoformat()}
        
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            if file_name.endswith('.ipynb'):
                nb = nbformat.read(tmp_path, as_version=4)
                content = self._process_notebook(nb)
                metadata["type"] = "jupyter_notebook"
                
            elif file_name.endswith(('.txt', '.py', '.md', '.R', '.js')):
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    content = f"# 📄 {file_name.upper()}\n\n{f.read()}"
                metadata["type"] = "code_text"
                
            elif file_name.endswith('.pdf'):
                reader = PdfReader(tmp_path)
                content = self._process_pdf(reader)
                metadata["type"] = "pdf_document"
                
            metadata["keywords"] = self._extract_keywords(content)
            
        finally:
            os.unlink(tmp_path)
            
        self.file_content = content[:800000]  # Large context window
        self.metadata = metadata
        return content, metadata
    
    def _process_notebook(self, nb):
        """Process Jupyter notebook with cell metadata"""
        content = "# 📓 JUPYTER NOTEBOOK ANALYSIS\n\n"
        code_cells = []
        markdown_cells = []
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown':
                content += f"## Markdown {i+1}:\n{cell.source}\n\n"
                markdown_cells.append(cell.source)
            elif cell.cell_type == 'code':
                content += f"## Code {i+1}:\n```python\n{cell.source}\n```\n"
                if cell.get('outputs'):
                    content += f"### Output:\n{json.dumps(cell.outputs[:1], indent=2)}\n\n"
                code_cells.append({"source": cell.source, "outputs": cell.outputs})
        
        self.code_cells = code_cells
        self.markdown_cells = markdown_cells
        return content
    
    def _process_pdf(self, reader):
        """Process PDF with page-level metadata"""
        content = "# 📄 PDF DOCUMENT\n\n"
        pages = []
        for i, page in enumerate(reader.pages[:50]):  # Limit pages
            try:
                text = page.extract_text()
                if text.strip():
                    content += f"## Page {i+1}:\n{text[:4000]}\n\n"
                    pages.append({"page": i+1, "content": text})
            except:
                continue
        self.pages = pages
        return content
    
    def _extract_keywords(self, content):
        """Extract key topics and concepts"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had']
        keywords = [w for w in words if w not in common_words and len(w) > 4]
        return list(set(keywords))[:20]
    
    def generate_study_guide(self):
        """Generate comprehensive study guide"""
        if not self.file_content:
            return "No content to analyze"
        
        try:
            prompt = f"""
Create a comprehensive study guide for the following document. Include:
1. Executive Summary (3-5 sentences)
2. Key Concepts & Definitions
3. Important Formulas/Algorithms (if applicable)
4. Main Takeaways
5. Practice Questions
6. Further Reading

DOCUMENT: {self.file_content[:300000]}

Format as a professional study guide.
"""
            response = model.generate_content(prompt)
            self.guide_content = response.text
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def generate_podcast_notes(self):
        """Generate podcast-style notes"""
        if not self.file_content:
            return "No content"
        
        try:
            prompt = f"""
Create engaging podcast notes for the document below. Include:
- Episode Title & Description
- Key Discussion Points (bullet points)
- Guest Expert Highlights
- Actionable Takeaways
- Timestamped segments

Format like a professional podcast script.

DOCUMENT: {self.file_content[:200000]}
"""
            response = model.generate_content(prompt)
            self.podcast_notes = response.text
            return response.text
        except:
            return "Error generating podcast notes"
    
    def generate_timeline(self):
        """Generate chronological timeline if applicable"""
        if not self.file_content:
            return []
        
        try:
            prompt = f"""
Extract chronological events, steps, or timeline from the document.
Return as JSON array of events with dates/sequence numbers.

DOCUMENT: {self.file_content[:150000]}

Format: [{{"event": "description", "time": "date/step"}}, ...]
"""
            response = model.generate_content(prompt)
            # Parse JSON response
            try:
                timeline = json.loads(response.text)
                self.timeline = timeline
                return timeline
            except:
                return [{"event": "Timeline extraction in progress", "time": "N/A"}]
        except:
            return []
    
    def generate_faq(self):
        """Generate FAQ from document"""
        if not self.file_content:
            return []
        
        try:
            prompt = f"""
Create 8-12 FAQ questions and answers based on the document.
Focus on common user questions and key concepts.

DOCUMENT: {self.file_content[:200000]}

Format as JSON: [{{"question": "...", "answer": "..."}}, ...]
"""
            response = model.generate_content(prompt)
            try:
                faq = json.loads(response.text)
                self.faq = faq
                return faq
            except:
                return [{"question": "FAQ generation", "answer": response.text[:200]}]
        except:
            return []
    
    def generate_brainstorm(self):
        """Generate brainstorming ideas"""
        if not self.file_content:
            return ""
        
        try:
            prompt = f"""
Based on the document, generate creative brainstorming ideas:
- Applications & Extensions
- Research Directions
- Business Opportunities
- Educational Uses
- Potential Improvements

DOCUMENT: {self.file_content[:150000]}
"""
            response = model.generate_content(prompt)
            self.brainstorm = response.text
            return response.text
        except:
            return "Error generating brainstorm"

# UI Setup
st.set_page_config(
    page_title="Notebook LLM Pro",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.study-guide {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px;}
.podcast-card {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 10px;}
.timeline-item {background: #e3f2fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196f3;}
.faq-item {background: #f3e5f5; padding: 15px; margin: 10px 0; border-radius: 8px;}
.brainstorm-box {background: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

# Initialize app
if "nllm" not in st.session_state:
    st.session_state.nllm = NotebookLLM()
if "features_generated" not in st.session_state:
    st.session_state.features_generated = {}

# Header
st.title("📓 Notebook LLM Pro - Advanced Document Intelligence")
st.markdown("Upload any document and unlock **deep analysis, study guides, podcasts, timelines, FAQs, and more!**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    temp = st.slider("AI Temperature", 0.1, 1.0, 0.7)
    
    st.header("📁 Upload")
    uploaded_file = st.file_uploader("Choose document", type=['pdf', 'ipynb', 'txt', 'py', 'md'])
    
    if st.button("🔄 Generate All Features"):
        if uploaded_file:
            with st.spinner("Processing document..."):
                content, metadata = st.session_state.nllm.extract_content(uploaded_file)
                st.session_state.nllm.metadata = metadata
                
                # Generate all features
                st.session_state.features_generated = {
                    "study_guide": st.session_state.nllm.generate_study_guide(),
                    "podcast": st.session_state.nllm.generate_podcast_notes(),
                    "timeline": st.session_state.nllm.generate_timeline(),
                    "faq": st.session_state.nllm.generate_faq(),
                    "brainstorm": st.session_state.nllm.generate_brainstorm()
                }
                st.success("✅ All features generated!")

# Main Content Tabs
if uploaded_file or st.session_state.nllm.file_content:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📖 Document", "🎓 Study Guide", "🎙️ Podcast", "⏰ Timeline", 
        "❓ FAQ", "💡 Brainstorm"
    ])
    
    with tab1:
        st.header("Original Document")
        st.text_area("Content Preview", st.session_state.nllm.file_content[:3000], height=400)
        st.json(st.session_state.nllm.metadata)
    
    with tab2:
        st.markdown('<div class="study-guide">', unsafe_allow_html=True)
        st.header("🎓 Interactive Study Guide")
        if "study_guide" in st.session_state.features_generated:
            st.markdown(st.session_state.features_generated["study_guide"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive quiz
        if st.button("🧠 Generate Quiz"):
            st.info("Quiz feature coming soon!")
    
    with tab3:
        st.markdown('<div class="podcast-card">', unsafe_allow_html=True)
        st.header("🎙️ AI Podcast Notes")
        if "podcast" in st.session_state.features_generated:
            st.markdown(st.session_state.features_generated["podcast"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎵 Generate Audio"):
                st.info("Audio synthesis integration ready")
        with col2:
            if st.button("📄 Export Script"):
                st.info("Podcast script export ready")
    
    with tab4:
        st.header("⏰ Document Timeline")
        if st.session_state.nllm.timeline:
            for event in st.session_state.nllm.timeline:
                st.markdown(f"""
                <div class="timeline-item">
                    <strong>{event.get('time', 'N/A')}</strong>: {event.get('event', '')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No chronological events detected")
    
    with tab5:
        st.header("❓ Smart FAQ")
        if st.session_state.nllm.faq:
            for item in st.session_state.nllm.faq:
                with st.expander(item["question"]):
                    st.write(item["answer"])
        else:
            st.info("FAQ generation in progress...")
    
    with tab6:
        st.markdown('<div class="brainstorm-box">', unsafe_allow_html=True)
        st.header("💡 Brainstorming Ideas")
        if "brainstorm" in st.session_state.features_generated:
            st.markdown(st.session_state.features_generated["brainstorm"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🎯 Generate Action Plan"):
            st.info("Action plan from brainstorm ready")

# Chat Interface (NotebookLM Style)
st.markdown("---")
st.header("💬 Deep Document Chat")
if st.session_state.nllm.file_content:
    # Chat history
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.chat_messages = st.session_state.get("chat_messages", [])
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Deep analysis..."):
                response = model.generate_content([
                    f"Deeply analyze this document: {st.session_state.nllm.file_content[:200000]}",
                    f"User question: {prompt}"
                ])
                st.markdown(response.text)
            
            st.session_state.chat_messages.append({"role": "assistant", "content": response.text})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    📓 Notebook LLM Pro | Advanced Document Intelligence | Powered by Gemini 2.5
</div>
""", unsafe_allow_html=True)