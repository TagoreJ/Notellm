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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gtts
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Safe API key loading
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY required!")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,  # Low for concise answers
            top_p=0.8,
            max_output_tokens=500  # Concise responses
        )
    )
    st.sidebar.success("✅ Fast NotebookLM Ready!")
except Exception as e:
    logger.error(f"API Error: {e}")
    st.error("❌ API Connection Failed!")
    st.stop()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def safe_hash(content):
    """Safe content hashing"""
    return hashlib.md5(content.encode()).hexdigest()

class FastNotebookLM:
    def __init__(self):
        self.chunks = []
        self.rag_index = None
        self.metadata = {}
        self.summary = ""
        self.is_processed = False
        self.progress = 0
        
    def update_progress(self, step, total_steps):
        """Update processing progress"""
        self.progress = (step / total_steps) * 100
        if hasattr(st.session_state, 'progress_bar'):
            st.session_state.progress_bar.progress(self.progress)
    
    def chunk_document(self, content, chunk_size=800, overlap=100):
        """Fast semantic chunking"""
        try:
            sentences = re.split(r'(?<=[.!?])\s+', content)
            chunks = []
            for i in range(0, len(sentences), chunk_size//100):
                chunk = ' '.join(sentences[i:i+chunk_size//100])
                if len(chunk) > 50:
                    chunks.append(chunk[:2000])  # Limit chunk size
            self.chunks = chunks[:50]  # Limit total chunks for speed
            return chunks
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            return [content[:5000]]
    
    def build_fast_rag(self):
        """Fast TF-IDF RAG index"""
        try:
            if len(self.chunks) < 2:
                return None
            vectorizer = TfidfVectorizer(
                max_features=2000, 
                stop_words='english',
                ngram_range=(1,2),
                lowercase=True
            )
            embeddings = vectorizer.fit_transform(self.chunks)
            self.rag_index = {'vectorizer': vectorizer, 'embeddings': embeddings}
            return self.rag_index
        except Exception as e:
            logger.error(f"RAG build error: {e}")
            return None
    
    def search_rag(self, query, top_k=3):
        """Fast RAG search"""
        try:
            if not self.rag_index or len(self.chunks) == 0:
                return [{'content': self.chunks[0][:500] if self.chunks else 'No content', 'score': 1.0}]
            
            vectorizer = self.rag_index['vectorizer']
            embeddings = self.rag_index['embeddings']
            
            query_vec = vectorizer.transform([query.lower()])
            similarities = cosine_similarity(query_vec, embeddings).flatten()
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Relevance threshold
                    results.append({
                        'content': self.chunks[idx],
                        'score': float(similarities[idx])
                    })
            return results if results else [{'content': self.chunks[0][:500], 'score': 0.5}]
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return [{'content': 'Search temporarily unavailable', 'score': 0}]
    
    def extract_content(self, uploaded_file):
        """Fast, error-proof content extraction"""
        try:
            self.progress = 0
            file_name = uploaded_file.name
            content = ""
            
            # Step 1: Save file
            self.update_progress(1, 5)
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            # Step 2: Extract based on type
            self.update_progress(2, 5)
            if file_name.lower().endswith('.ipynb'):
                content = self._extract_notebook(tmp_path)
            elif file_name.lower().endswith('.pdf'):
                content = self._extract_pdf(tmp_path)
            else:  # Text files
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f"# 📄 {file_name}\n\n" + f.read()
            
            # Cleanup
            os.unlink(tmp_path)
            
            # Step 3: Chunking
            self.update_progress(3, 5)
            self.chunk_document(content)
            
            # Step 4: RAG indexing
            self.update_progress(4, 5)
            self.build_fast_rag()
            
            self.metadata = {
                'filename': file_name,
                'chunks': len(self.chunks),
                'size': len(content),
                'processed_at': datetime.now().isoformat()
            }
            
            self.is_processed = True
            self.update_progress(5, 5)
            return True
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            st.error(f"Processing failed: {str(e)}")
            return False
    
    def _extract_notebook(self, path):
        """Extract notebook content"""
        try:
            nb = nbformat.read(path, as_version=4)
            content = "# 📓 JUPYTER NOTEBOOK\n"
            for i, cell in enumerate(nb.cells[:15]):  # Limit cells
                if cell.cell_type == 'code':
                    content += f"\n## Code {i+1}\n```{cell.source[:1000]}\n```\n"
                else:
                    content += f"\n## {cell.source[:500]}\n"
            return content
        except:
            return "# Notebook extraction failed"
    
    def _extract_pdf(self, path):
        """Extract PDF content"""
        try:
            reader = PdfReader(path)
            content = "# 📄 PDF DOCUMENT\n"
            for i, page in enumerate(reader.pages[:20]):  # Limit pages
                try:
                    text = page.extract_text()
                    if text and len(text.strip()) > 10:
                        content += f"\n## Page {i+1}\n{text[:1500]}\n"
                except:
                    continue
            return content
        except:
            return "# PDF extraction failed"
    
    def generate_concise_summary(self):
        """Fast summary generation"""
        try:
            if not self.chunks:
                return "No content to summarize"
            
            context = ' '.join(self.chunks[:5])  # Use top chunks
            prompt = f"""
            Provide a CONCISE summary (3-5 sentences max) of this document:
            Focus on MAIN POINTS only.
            {context}
            """
            
            response = model.generate_content(prompt)
            self.summary = response.text
            return response.text
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return "Summary generation failed"
    
    def generate_fast_response(self, query):
        """Fast, relevant RAG-powered response"""
        try:
            # Get relevant context
            relevant = self.search_rag(query, top_k=2)
            context = '\n'.join([r['content'][:800] for r in relevant])
            
            prompt = f"""
            Answer this question using ONLY the provided context:
            Be CONCISE and RELEVANT. Max 3-4 sentences.
            
            RELEVANT CONTEXT:
            {context}
            
            QUESTION: {query}
            
            Direct, focused answer:
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Response error: {e}")
            return "Sorry, I couldn't process that request."
    
    def generate_audio_fast(self, text):
        """Fast audio generation"""
        try:
            if len(text) > 300:  # Limit for speed
                text = text[:300]
            
            tts = gtts.gTTS(text, lang='en', slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes.getvalue()
        except Exception as e:
            logger.error(f"Audio error: {e}")
            return None
    
    def search_images_fast(self, query):
        """Fast image search with Unsplash source"""
        try:
            # Using Unsplash source URLs (no API key needed)
            base_url = "https://source.unsplash.com"
            search_query = query.replace(' ', '+')[:20]  # Limit query length
            image_urls = [
                f"{base_url}/400x300/?{search_query}",
                f"{base_url}/400x300/?{search_query},document",
                f"{base_url}/400x300/?{search_query},study"
            ]
            return image_urls[:2]  # Limit for speed
        except:
            return ["https://via.placeholder.com/400x300?text=Image"]
    
    def fetch_image_safe(self, url):
        """Safe image fetching with timeout"""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img.resize((300, 200), Image.Resampling.LANCZOS)
        except:
            pass
        return None

# Streamlit App
def main():
    st.set_page_config(
        page_title="Fast NotebookLM", 
        page_icon="⚡", 
        layout="wide"
    )
    
    # Custom CSS for fast, clean UI
    st.markdown("""
    <style>
    .fast-header {background: linear-gradient(135deg, #00c6ff, #0072ff); color: white; padding: 2rem; border-radius: 15px; text-align: center;}
    .feature-card {background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0;}
    .rag-result {background: #e3f2fd; padding: 0.8rem; border-radius: 5px; margin: 0.3rem 0; font-size: 0.9em;}
    .concise-answer {background: #e8f5e8; padding: 1rem; border-radius: 8px; border-left: 4px solid #4caf50;}
    .progress-container {background: #f0f0f0; padding: 1rem; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="fast-header"><h1>⚡ Fast NotebookLM</h1><p>Instant document insights with RAG & progress tracking</p></div>', unsafe_allow_html=True)
    
    # Initialize session
    if "nllm" not in st.session_state:
        st.session_state.nllm = FastNotebookLM()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload")
        uploaded_file = st.file_uploader("Choose document", type=['pdf', 'ipynb', 'txt', 'py', 'md'])
        
        if uploaded_file and st.button("🚀 Process Fast", use_container_width=True):
            with st.spinner("Processing..."):
                # Progress bar
                progress_bar = st.progress(0)
                st.session_state.progress_bar = progress_bar
                
                success = st.session_state.nllm.extract_content(uploaded_file)
                if success:
                    st.success(f"✅ Processed {st.session_state.nllm.metadata.get('chunks', 0)} chunks")
                    st.session_state.processed = True
                else:
                    st.error("Processing failed")
    
    # Main tabs
    if hasattr(st.session_state, 'processed') and st.session_state.processed:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 RAG Search", "💬 Fast Chat", "🎵 Audio"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.header("📋 Instant Summary")
            if st.button("✨ Generate Summary"):
                with st.spinner("Summarizing..."):
                    summary = st.session_state.nllm.generate_concise_summary()
                    st.markdown(f'<div class="concise-answer"><strong>Summary:</strong> {summary}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1: st.metric("📄 Chunks", st.session_state.nllm.metadata.get('chunks', 0))
            with col2: st.metric("⚡ Status", "RAG Ready")
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.header("🔍 Lightning RAG Search")
            query = st.text_input("Search document...", key="rag_query")
            
            if query or st.button("🔎 Search"):
                with st.spinner("Searching..."):
                    results = st.session_state.nllm.search_rag(query)
                    for i, result in enumerate(results):
                        st.markdown(f"""
                        <div class="rag-result">
                        <strong>#{i+1} ({result['score']:.1%})</strong><br>
                        {result['content'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.header("💬 Fast AI Chat")
            
            # Chat history
            for msg in st.session_state.chat_history[-6:]:  # Last 6 messages
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about document..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking fast..."):
                        answer = st.session_state.nllm.generate_fast_response(prompt)
                        st.markdown(f'<div class="concise-answer">{answer}</div>', unsafe_allow_html=True)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.header("🎵 Quick Audio")
            text = st.text_area("Text to speak", height=80, placeholder="Or use summary...")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔊 Generate Audio"):
                    with st.spinner("Creating audio..."):
                        audio_data = st.session_state.nllm.generate_audio_fast(text or st.session_state.nllm.summary)
                        if audio_data:
                            st.audio(audio_data)
                            st.success("Audio ready!")
                        else:
                            st.warning("Audio generation failed")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.info("""
        🚀 **Fast NotebookLM Features:**
        • ⚡ **Lightning Processing** - Progress bars + caching
        • 🔍 **Smart RAG** - Semantic document search  
        • 💬 **Concise Answers** - No fluff, just relevant info
        • 🎵 **Instant Audio** - Text-to-speech summaries
        • 🛡️ **Error-Proof** - Safe extraction & fallbacks
        
        **Upload your document to start!** (PDF, notebooks, code, text)
        """)
    
    # Global controls
    if st.sidebar.button("🗑️ Clear All", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['nllm']:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()