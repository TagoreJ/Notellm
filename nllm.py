import streamlit as st
import google.generativeai as genai
import nbformat
from PyPDF2 import PdfReader
import os
import tempfile
import json
from pathlib import Path
import pydot
from io import BytesIO

# ================= Streamlit Cloud Secret Loading =================
# Use secrets.toml for GEMINI_API_KEY
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ GEMINI_API_KEY missing in Streamlit secrets.")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]

# ================= Initialize Gemini Model =================
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    st.sidebar.success("✅ Gemini 2.0 Connected via Streamlit Secrets!")
except Exception as e:
    st.error(f"❌ Gemini API Error: {e}")
    st.info("💡 Update SDK: pip install --upgrade google-generativeai")
    st.stop()

# ================= File Extraction =================
def extract_notebook_content(nb):
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
        return content
    except Exception as e:
        return f"Error parsing notebook: {str(e)}"

def extract_text_content(file_path, file_name):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"# 📄 {Path(file_name).stem.upper()}\n\n{content}"
    except:
        return "Error reading text file"

def extract_pdf_content(file_path):
    try:
        reader = PdfReader(file_path)
        content = "# 📄 PDF DOCUMENT\n\n"
        for i, page in enumerate(reader.pages[:10], 1):
            text = page.extract_text()
            if text and text.strip():
                content += f"## Page {i}\n{text[:2000]}\n\n"
        return content if content != "# 📄 PDF DOCUMENT\n\n" else "No text extracted from PDF"
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"

@st.cache_data
def process_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    file_content = uploaded_file.read()
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    try:
        if file_name.endswith(".ipynb"):
            nb = nbformat.read(tmp_path, as_version=4)
            return extract_notebook_content(nb)
        elif file_name.endswith((".txt", ".py", ".md", ".r", ".js", ".cpp", ".java", ".json")):
            return extract_text_content(tmp_path, file_name)
        elif file_name.endswith(".pdf"):
            return extract_pdf_content(tmp_path)
        else:
            return f"Unsupported file: {file_name}"
    finally:
        os.unlink(tmp_path)

# ================= Mind Map Generators =================
def create_mind_map_from_notebook(nb):
    graph = pydot.Dot(graph_type="graph", rankdir="LR")
    root = pydot.Node("Notebook", style="filled", fillcolor="lightblue", shape="box")
    graph.add_node(root)

    for i, cell in enumerate(nb.cells, 1):
        if cell.cell_type == "markdown":
            label = cell.source.strip().split("\n")[0][:40]
            node = pydot.Node(f"MD {i}: {label}", shape="note")
        elif cell.cell_type == "code":
            node = pydot.Node(f"Code Cell {i}", shape="box", style="filled", fillcolor="lightgrey")
        else:
            continue
        graph.add_node(node)
        graph.add_edge(pydot.Edge(root, node))
    return graph.to_string()

def create_mind_map_from_text(text_content):
    graph = pydot.Dot(graph_type="graph", rankdir="LR")
    root = pydot.Node("Document", style="filled", fillcolor="lightgreen", shape="box")
    graph.add_node(root)

    lines = text_content.split("\n")
    count = 0
    for line in lines:
        line = line.strip()
        if line and (line.startswith("#") or len(line) < 80 and len(line.split()) < 8):
            label = line[:40]
            node = pydot.Node(f"{label}_{count}", label=label, shape="ellipse")
            graph.add_node(node)
            graph.add_edge(pydot.Edge(root, node))
            count += 1
            if count > 15:
                break
    return graph.to_string()

def render_mind_map(graph_str):
    graphs = pydot.graph_from_dot_data(graph_str)
    if graphs:
        graph = graphs[0]
        return graph.create_png()
    return None

# ================= AI Response =================
def generate_ai_response(prompt, context):
    try:
        system_prompt = f"""
You are a helpful document analysis assistant.
Use the content below to answer questions with structure and clarity.

CONTEXT:
{context}

QUESTION:
{prompt}
"""
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# ================= Streamlit UI =================
st.set_page_config(
    page_title="Notebook LLM + Mind Map",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📓 Notebook LLM + 🧠 Mind Map (Streamlit Cloud Ready)")
st.markdown("Chat with your notebooks, visualize key ideas with a Mind Map, and explore using Gemini 2.0!")

uploaded_file = st.file_uploader(
    "📁 Upload Document",
    type=["ipynb", "py", "txt", "md", "r", "js", "cpp", "java", "json", "pdf"],
    help="Upload your notebook or document"
)

if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mind_map_img" not in st.session_state:
    st.session_state.mind_map_img = None

if uploaded_file:
    file_name = uploaded_file.name.lower()
    with st.spinner("🔍 Reading and processing..."):
        content = process_uploaded_file(uploaded_file)
        st.session_state.file_content = content

        # Generate Mind Map
        if file_name.endswith(".ipynb"):
            nb = nbformat.reads(uploaded_file.getvalue().decode("utf-8"), as_version=4)
            mind_map_dot = create_mind_map_from_notebook(nb)
        else:
            mind_map_dot = create_mind_map_from_text(content)
        st.session_state.mind_map_img = render_mind_map(mind_map_dot)

    st.success(f"✅ Loaded: {uploaded_file.name}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 Chat with AI")
        for msg in st.session_state.messages[-8:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Ask something about your document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    ans = generate_ai_response(prompt, st.session_state.file_content)
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})

    with col2:
        st.subheader("🧠 Mind Map View")
        if st.session_state.mind_map_img:
            st.image(st.session_state.mind_map_img, use_column_width=True)
        else:
            st.info("Mind map appears here after processing.")

else:
    st.info("📂 Upload a `.ipynb`, `.pdf`, or `.py` file to start.")

st.markdown("---")
st.caption("🚀 Built for Streamlit Cloud | Gemini 2.0 | Updated October 2025")
