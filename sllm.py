import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# Load tokenizer and model once, cache it
@st.cache_resource(show_spinner=False)
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator

generator = load_local_llm()

st.set_page_config(page_title="Notebook LLM with Local DistilGPT2", layout="wide")

st.title("📓 Notebook LLM with Local LLM Mode")

# Sidebar model selection
mode = st.sidebar.selectbox("Select Model Mode", ["Gemini Cloud API", "Local DistilGPT2"])

if mode == "Local DistilGPT2":
    st.info("Running small local DistilGPT2 model inference in this session (no API key needed).")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("Chat with Local DistilGPT2")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Enter your question for Local LLM..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            progress_bar = st.progress(0)
            start_time = time.time()

            # Simulate progress in steps
            output_text = ""
            for i in range(1, 6):
                time.sleep(0.3)  # simulate chunk generation delay
                progress_bar.progress(i * 20)

            # Actual generation call (can be slow on CPU)
            outputs = generator(prompt, max_length=100, do_sample=True)
            output_text = outputs[0]['generated_text']

            elapsed = time.time() - start_time
            progress_bar.progress(100)  # complete progress bar

            st.markdown(output_text)
            st.caption(f"⏳ Generated in {elapsed:.2f} seconds")

            st.session_state.messages.append({"role": "assistant", "content": output_text})

else:
    st.warning("Gemini API mode not implemented here. Use your existing integration.")

