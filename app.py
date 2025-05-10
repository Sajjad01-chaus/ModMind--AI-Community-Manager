import os
import streamlit as st
from dotenv import load_dotenv
from agent import ingest_knowledge, get_answer

load_dotenv()
st.set_page_config(page_title="AI Community Moderator", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 ModMind: Your AI Community Moderator</h1>", unsafe_allow_html=True)

state = st.session_state
state.setdefault("history", [])
state.setdefault("vectorstore", None)
state.setdefault("brand_name", "")
state.setdefault("brand_voice", "")
state.setdefault("model_name", "llama3-8b-8192")
state.setdefault("temperature", 0.2)
state.setdefault("retrieval_k", 5)
state.setdefault("page", "chat")

# sidebar 
with st.sidebar:
    st.title("ModMind: AI Community Moderator")
    if st.button("💬 Chat", type="primary" if state.page=="chat" else "secondary"):
        state.page = "chat"
    if st.button("📚 Knowledge", type="primary" if state.page=="knowledge" else "secondary"):
        state.page = "knowledge"
    if st.button("⚙️ Settings", type="primary" if state.page=="settings" else "secondary"):
        state.page = "settings"

# Chat tab
if state.page == "chat":
    st.header("💬 Community Moderator Chat")
    if not state.vectorstore:
        st.warning("Add knowledge sources under 📚 Knowledge first.")
    for msg in state.history:
        who = "You" if msg['role']=="user" else "Moderator"
        st.chat_message(msg['role']).write(f"**{who}:** {msg['content']}")
    q = st.chat_input("Ask a question…")
    if q:
        state.history.append({"role":"user","content":q})
        if not state.vectorstore:
            ans = "No knowledge base!"
        else:
            ans = get_answer(
                state.vectorstore,
                state.brand_name,
                state.brand_voice,
                q,
                state.model_name,
                state.temperature,
                state.retrieval_k
            )
        state.history.append({"role":"assistant","content":ans})
        st.chat_message("assistant").write(ans)

# knowledge tab
elif state.page == "knowledge":
    st.header("📚 Knowledge Management")
    with st.expander("🔑 Brand Settings", expanded=True):
        state.brand_name  = st.text_input("Brand Name", state.brand_name)
        state.brand_voice = st.text_area("Brand Voice", state.brand_voice, height=120)
    with st.expander("🌐 Data Sources", expanded=True):
        urls  = st.text_area("URLs (one per line)").splitlines()
        files = st.file_uploader("Upload documents", type=["pdf","txt","docx"], accept_multiple_files=True)
        if st.button("Process Sources"):
            state.vectorstore = ingest_knowledge(urls, files)
            total = state.vectorstore.index.ntotal
            st.success(f"Built with {total} chunks indexed.")
        if state.vectorstore:
            st.write("**Current Sources:**")
            for u in urls:  st.write("-", u)
            for f in files: st.write("-", f.name)
            if st.button("Clear Knowledge Base"):
                state.vectorstore = None
                state.history.clear()
                st.success("Cleared.")

# Settings tab
else:
    st.header("⚙️ System Settings")
    st.subheader("Brand Settings")
    state.brand_name  = st.text_input("Brand Name", state.brand_name)
    state.brand_voice = st.text_area("Brand Voice Guidelines", state.brand_voice, height=100)

    st.subheader("Model Configuration")
    state.model_name = st.selectbox(
        "LLM Model",
        ["llama3-8b-8192","llama3-70b-8192","mixtral-8x7b-32768"],
        index=["llama3-8b-8192","llama3-70b-8192","mixtral-8x7b-32768"].index(state.model_name)
    )
    state.temperature = st.slider("Temperature", 0.0, 1.0, state.temperature, step=0.05)
    state.retrieval_k  = st.slider("Top-k docs", 1, 10, state.retrieval_k)
    if st.button("Save Settings"):
        st.success("Settings updated!")
