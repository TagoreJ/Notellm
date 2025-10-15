import streamlit as st
import google.generativeai as genai
import nbformat
from PyPDF2 import PdfReader
import os
import tempfile
import json
from io import BytesIO
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import time
import re

# Load environment variables
load_dotenv()

# Configuration - Works for Streamlit Cloud + Local
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found!")
    st.markdown("""
    **Local:** Add to `.env` file: `GEMINI_API_KEY=your_key`  
    **Streamlit Cloud:** Settings → Secrets → Add `GEMINI_API_KEY = your_key`
    """)
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    st.sidebar.success("✅ Gemini 2.5 Connected!")
except Exception as e:
    st.error(f"❌ API Error: {str(e)}")
    st.stop()

# Dynamic image sources based on content type
def get_relevant_images(topic_keywords):
    """Get relevant images based on document content"""
    images = {
        # Ayurveda/Health
        "ayurveda": [
            "https://cdn.pixabay.com/photo/2017/03/08/12/16/ayurveda-2128935_1280.jpg",
            "https://images.unsplash.com/photo-1564103146797-fd5d731e63c1?auto=format&fit=crop&w=800",
        ],
        "health": [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=800",
            "https://cdn.pixabay.com/photo/2016/11/29/05/14/yoga-1868556_1280.jpg",
        ],
        # Tech/Code
        "python": [
            "https://images.unsplash.com/photo-1526379095098-d400c8895b16?auto=format&fit=crop&w=800",
            "https://cdn.pixabay.com/photo/2017/08/07/14/02/code-2606586_1280.jpg",
        ],
        "code": [
            "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=800",
            "https://cdn.pixabay.com/photo/2017/08/07/14/02/code-2606585_1280.jpg",
        ],
        # Default/Generic
        "default": [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=800",
            "https://cdn.pixabay.com/photo/2017/08/07/14/02/code-2606586_1280.jpg",
        ]
    }
    
    for key in topic_keywords:
        if key in images:
            return images[key]
    return images["default"]

def fetch_image(url, max_size=(400, 300)):
    """Fetch and resize image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img = img.resize(max_size, Image.Resampling.LANCZOS)
        return img
    except:
        return None

def extract_text_from_file(uploaded_file):
    """Extract content from any file type"""
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
        
        elif file_name.endswith(('.txt', '.py', '.md', '.R', '.js', '.cpp', '.java', '.json')):
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                content = f"# 📄 {file_name.upper()}\n\n{f.read()}"
        
        elif file_name.endswith('.pdf'):
            reader = PdfReader(tmp_file_path)
            content = "# 📄 PDF DOCUMENT\n\n"
            for i, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        content += f"## Page {i}\n{text[:3000]}\n\n"
                except:
                    continue
        
        else:
            return f"Unsupported file type: {file_name}"
    
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    return content[:500000]

def detect_document_topic(content):
    """Simple topic detection for relevant images"""
    content_lower = content.lower()
    keywords = {
        "ayurveda": ["ayurveda", "dosha", "vata", "pitta", "kapha", "herbs"],
        "health": ["health", "wellness", "yoga", "meditation", "nutrition"],
        "python": ["python", "import", "def ", "class ", "pandas", "numpy"],
        "code": ["function", "class", "def ", "import", "algorithm"]
    }
    
    for topic, words in keywords.items():
        if any(word in content_lower for word in words):
            return [topic]
    return ["default"]

def generate_response(prompt, file_content, temperature=0.7):
    """Generate comprehensive response with deep understanding"""
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.9,
                max_output_tokens=2000
            )
        )
        
        system_prompt = f"""
You are an expert document analysis assistant with deep understanding of the uploaded content.
You have thoroughly analyzed and memorized the entire document below. Answer ANY question based EXCLUSIVELY on this content.

Be comprehensive, structured, and helpful. Reference specific sections when possible.
For health/ayurveda documents: Provide practical tips and advice.
For code: Explain functionality, suggest improvements.
For research: Summarize findings, explain methodology.

FULL DOCUMENT CONTENT (use as your complete knowledge base):
{file_content}

USER QUESTION: {prompt}

Provide a detailed, knowledgeable response based only on the document above.
"""
        
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_text_mind_map(content, max_nodes=8):
    """Cloud-friendly text-based mind map"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    key_phrases = []
    
    # Extract key phrases (simple heuristic)
    for line in lines[:20]:  # First 20 lines for main concepts
        words = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', line)
        if words:
            key_phrases.extend(words[:2])
    
    unique_phrases = list(set(key_phrases))[:max_nodes]
    
    mind_map = f"""
🌐 **DOCUMENT MIND MAP**
╔══════════════════════════════════════╗
║          📚 MAIN DOCUMENT            ║
╠══════════════════════╦═══════════════╣
║                      ║               ║
║  ├─ {unique_phrases[0] if len(unique_phrases)>0 else 'Concept 1'}    ║
║  ├─ {unique_phrases[1] if len(unique_phrases)>1 else 'Concept 2'}    ║
║  ├─ {unique_phrases[2] if len(unique_phrases)>2 else 'Concept 3'}    ║
║  └─ {'...' if len(unique_phrases)>3 else 'More Concepts'}           ║
╚══════════════════════╩═══════════════╝

💡 Key concepts automatically extracted from your document
"""
    return mind_map

# Professional UI Setup (FIXED - No duplicate page_title)
st.set_page_config(
    page_title="Document LLM Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Design
st.markdown("""
<style>
    .main-header {color: #1E88E5; font-family: 'Georgia', serif; text-align: center; font-size: 2.5em;}
    .sub-header {color: #1976D2; font-family: 'Arial', sans-serif;}
    .chat-bubble-user {background: linear-gradient(135deg, #E3F2FD, #BBDEFB); border-radius: 15px; padding: 15px; margin: 10px 0;}
    .chat-bubble-assistant {background: linear-gradient(135deg, #E8F5E8, #C8E6C9); border-radius: 15px; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50;}
    .response-section {background: #F5F5F5; border-left: 5px solid #2196F3; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .mindmap-box {background: #FFF3E0; border: 2px dashed #FF9800; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace;}
    .image-gallery {display: flex; justify-content: space-around; flex-wrap: wrap;}
    .metric-card {background: white; padding: 1rem; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 Document LLM Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2em;">Upload any document and unlock deep AI insights</p>', unsafe_allow_html=True)

# Sidebar - Professional Controls
with st.sidebar:
    st.header("⚙️ AI Settings")
    temperature = st.slider("🤖 Creativity Level", 0.1, 1.0, 0.7, 0.1)
    st.info(f"Current: {temperature:.1f} - Higher = More creative analysis")
    
    st.markdown("---")
    st.header("📁 Supported Formats")
    st.markdown("• 📖 **PDF Books/Documents**")
    st.markdown("• 💻 **Code** (.py, .js, .cpp, .java)")
    st.markdown("• 📝 **Text** (.txt, .md)")
    st.markdown("• 📓 **Jupyter** (.ipynb)")
    
    st.markdown("---")
    st.info("🆓 **Powered by:** Gemini 2.5 Flash")
    if st.button("🔑 API Setup", use_container_width=True):
        st.markdown("[Get Free Key](https://aistudio.google.com/app/apikey)")

# Main Content Layout
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 Upload Your Document",
        type=['pdf', 'txt', 'md', 'ipynb', 'py', 'js', 'cpp', 'java'],
        help="Upload any document for AI analysis"
    )

with col2:
    st.markdown('<h3 class="sub-header">💡 Example Queries</h3>', unsafe_allow_html=True)
    examples = [
        "Summarize the main points",
        "Explain key concepts",
        "Provide practical tips",
        "What are the main findings?",
        "Debug/analyze the code"
    ]
    for example in examples:
        st.markdown(f"• {example}")

# Session State
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = ""
if "topic" not in st.session_state:
    st.session_state.topic = "default"

# Process File
if uploaded_file is not None:
    with st.spinner("🔄 Analyzing document content..."):
        content = extract_text_from_file(uploaded_file)
        st.session_state.file_content = content
        st.session_state.filename = uploaded_file.name
        st.session_state.topic = detect_document_topic(content)[0]
    
    if not content.startswith(("Error", "Unsupported")):
        # Success Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document", st.session_state.filename)
        with col2:
            st.metric("📊 Content Size", f"{len(content)/1000:.1f} KB")
        with col3:
            st.metric("🎯 Topic", st.session_state.topic.upper())
        
        # Preview
        with st.expander("👁️ Document Preview", expanded=False):
            st.text_area("Content", content[:2000], height=200, disabled=True)
        
        # Mind Map Option
        if st.checkbox("🧠 Generate Mind Map"):
            mind_map = create_text_mind_map(content)
            st.markdown('<div class="mindmap-box">', unsafe_allow_html=True)
            st.markdown(mind_map)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat Interface
        st.markdown('<h2 class="sub-header">💬 AI Document Chat</h2>', unsafe_allow_html=True)
        
        # Chat History
        for message in st.session_state.messages[-10:]:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-assistant">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat Input
        if prompt := st.chat_input("Ask about your document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI is analyzing your document..."):
                    response = generate_response(prompt, st.session_state.file_content, temperature)
                    st.markdown(f'<div class="response-section">{response}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Dynamic Relevant Images
                img_urls = get_relevant_images([st.session_state.topic])
                st.markdown('<h4>🖼️ Visual Context</h4>', unsafe_allow_html=True)
                cols = st.columns(len(img_urls))
                for i, url in enumerate(img_urls):
                    with cols[i]:
                        img = fetch_image(url)
                        if img:
                            st.image(img, use_column_width=True)
                        time.sleep(0.2)  # Rate limiting
        
        # Controls
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_btn2:
            if st.button("📁 New Document", use_container_width=True):
                st.session_state = {"file_content": "", "messages": [], "filename": "", "topic": "default"}
                st.rerun()
    else:
        st.error(f"❌ {content}")

# Welcome Screen
if uploaded_file is None and not st.session_state.messages:
    st.info("""
    🚀 **Welcome to Document LLM Assistant!**
    
    **Upload any document** (PDF, code, text, notebooks) and get:
    • Deep AI understanding of your content
    • Comprehensive answers to any question
    • Automatic topic detection
    • Visual context with relevant images
    • Professional mind maps
    
    **Works with:** Research papers, codebases, books, technical docs, health guides, and more!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    📚 Document LLM Assistant | Professional AI Document Analysis | Powered by Streamlit + Gemini 2.5
</div>
""", unsafe_allow_html=True)