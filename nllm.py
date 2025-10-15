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

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found! Add to `.env` (local) or app secrets (Streamlit Cloud).")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    st.sidebar.success("✅ Gemini 2.5 Connected!")
except Exception as e:
    st.error(f"❌ API Error: {str(e)}")
    st.stop()

# Ayurveda-themed image URLs (free from Pixabay/Unsplash)
HEALTH_IMAGES = [
    "https://cdn.pixabay.com/photo/2017/03/08/12/16/ayurveda-2128935_1280.jpg",  # Herbs
    "https://images.unsplash.com/photo-1564103146797-fd5d731e63c1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",  # Massage
    "https://cdn.pixabay.com/photo/2017/03/08/12/16/ayurveda-2128934_1280.jpg",  # Treatment
    "https://images.unsplash.com/photo-1580259087076-9df8c0a5a0e7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",  # Wellness
    "https://cdn.pixabay.com/photo/2017/03/08/12/16/ayurveda-2128933_1280.jpg",  # Medicine
    "https://images.unsplash.com/photo-1576092768241-dec231879fc3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",  # Herbs
    "https://cdn.pixabay.com/photo/2017/03/08/12/16/ayurveda-2128932_1280.jpg",  # Spa
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"   # Yoga
]

def fetch_image(url):
    """Fetch and return PIL Image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None

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
            for i, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        content += f"## Page {i}:\n{text[:3000]}\n\n"
                except:
                    continue
        
        else:
            return f"Unsupported file type: {file_name}"
    
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    return content[:500000]  # Increased for better book understanding

def generate_response(prompt, file_content, temperature=0.7):
    """Generate AI response using file context with temperature for creativity"""
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,  # Higher temp for more creative/understanding responses
                top_p=0.9,
                max_output_tokens=1500
            )
        )
        system_prompt = f"""
You are an expert Ayurveda and health advisor. You have thoroughly analyzed the uploaded book content below.
Understand the entire book deeply - treat it as your knowledge base. Answer ANY question based ONLY on this content.
Be comprehensive, helpful, and provide detailed tips, explanations, and advice as per the book's teachings.
If the question is about health tips, draw directly from the book's principles (doshas, herbs, routines, etc.).

FULL BOOK CONTENT (analyze deeply):
{file_content}

USER QUESTION: {prompt}

Respond knowledgeably, structured, and engagingly.
"""
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_mind_map(text):
    """Create simple mind map using text-based representation (works on Streamlit Cloud)"""
    # Simple keyword extraction and tree structure
    lines = text.split('\n')
    nodes = []
    for line in lines[:10]:  # Limit for simplicity
        if line.strip():
            nodes.append(line.strip()[:50] + '...')  # Truncate
    
    mind_map = """
    📖 BOOK MIND MAP
    ┌─────────────────┐
    │     ROOT        │  <- Main Topic
    └─────────┬───────┘
              │
    ├─ Branch 1 ── {0}
    │
    ├─ Branch 2 ── {1}
    │
    └─ Branch 3 ── {2}
    
    (Interactive JS mind map would require additional libs)
    """.format(*nodes[:3])
    
    return mind_map

# Professional UI Setup
st.set_page_config(
    page_title="Ayurveda Notebook LLM",
    page_title="Ayurveda Notebook LLM",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main-header {color: #2E7D32; font-family: 'serif'; text-align: center;}
    .chat-bubble {background: linear-gradient(135deg, #E8F5E8, #C8E6C9); border-radius: 15px; padding: 15px; margin: 10px 0;}
    .response-section {background: #F1F8E9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 10px;}
    .image-placeholder {text-align: center; margin: 10px;}
    .mindmap-box {background: #E3F2FD; border: 2px dashed #2196F3; padding: 15px; border-radius: 10px; font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌿 Ayurveda Notebook LLM - Health Wisdom Assistant</h1>', unsafe_allow_html=True)
st.markdown("Upload your Ayurveda book (PDF/TXT) and unlock personalized health insights with deep AI understanding!")

# Sidebar - Professional Controls
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("### Model Configuration")
    temperature = st.slider("AI Temperature (Creativity)", 0.1, 1.0, 0.7, 0.1)
    st.info(f"🌡️ Current: {temperature} - Higher = More creative responses from book")
    
    st.markdown("### Supported Formats")
    st.markdown("• 📖 PDF Books")
    st.markdown("• 📄 Text Files")
    st.markdown("• 📓 Jupyter Notes")
    
    st.markdown("---")
    st.info("🆓 Powered by Gemini 2.5 Flash")
    if st.button("🔑 API Setup"):
        st.markdown("[Get Free Key](https://aistudio.google.com/app/apikey)")

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📖 Upload Ayurveda Book",
        type=['pdf', 'txt', 'md', 'ipynb'],
        help="Upload your PDF book for deep analysis"
    )

with col2:
    st.markdown("### 💡 Quick Tips")
    st.markdown("• 'Dosha balancing tips'")
    st.markdown("• 'Daily health routine'")
    st.markdown("• 'Herbal remedies for digestion'")

# Session state
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = ""

# Process file
if uploaded_file is not None:
    with st.spinner("🔄 Deeply Analyzing Book Content..."):
        content = extract_text_from_file(uploaded_file)
        st.session_state.file_content = content
        st.session_state.filename = uploaded_file.name
    
    if not content.startswith(("Error", "Unsupported")):
        st.success(f"✅ Book Loaded: **{st.session_state.filename}**")
        
        # Preview
        with st.expander("📖 Book Preview", expanded=False):
            st.text_area("Excerpt", st.session_state.file_content[:2000], height=200, disabled=True)
        
        # Mind Map Option
        if st.checkbox("🧠 Generate Mind Map"):
            mind_map = create_mind_map(st.session_state.file_content)
            st.markdown('<div class="mindmap-box">', unsafe_allow_html=True)
            st.markdown(mind_map)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat Interface
        st.subheader("💬 Health Query Chat")
        
        # Chat History
        for message in st.session_state.messages[-10:]:
            with st.chat_message(message["role"]):
                st.markdown(f'<div class="chat-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Input
        if prompt := st.chat_input("Ask about health tips from your book..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-bubble">{prompt}</div>', unsafe_allow_html=True)
            
            # Response
            with st.chat_message("assistant"):
                with st.spinner("🌿 Consulting ancient wisdom..."):
                    response = generate_response(prompt, st.session_state.file_content, temperature)
                    st.markdown(f'<div class="response-section">{response}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Relevant Images
                st.markdown("### 🖼️ Visual Inspiration")
                img_urls = HEALTH_IMAGES[:3]  # Pick 3 relevant
                for url in img_urls:
                    img = fetch_image(url)
                    if img:
                        st.image(img, use_column_width=True, caption="Ayurveda Health Tip")
                    else:
                        st.markdown(f'<div class="image-placeholder">[Image: {url.split("/")[-1]}]</div>', unsafe_allow_html=True)
                        time.sleep(0.5)  # Rate limit
        
        # Controls
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col_btn2:
            if st.button("📚 New Book"):
                st.session_state.file_content = ""
                st.session_state.messages = []
                st.rerun()
    else:
        st.error(f"❌ {content}")

# Welcome
if uploaded_file is None and not st.session_state.messages:
    st.info("""
    🌿 **Welcome to Ayurveda Wisdom!**
    Upload your book, and I'll deeply understand its content to provide personalized health advice.
    No more 'I can't' - now it's 'Here's wisdom from your book'!
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #4CAF50;'>
        🌿 Ayurveda Notebook LLM | Professional Health Insights | Powered by Streamlit & Gemini 2.5
    </div>
    """, unsafe_allow_html=True
)