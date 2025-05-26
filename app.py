import streamlit as st
import os
import sys
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict

from graph_setup import build_rag_graph, build_graph, rag_runner

from langchain_core.tools import Tool
from langgraph.prebuilt.tool_node import ToolNode

import streamlit as st

def debug_log(line: str):
    """Append a debug line to session_state and show it in the sidebar."""
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(line)

    # Render in a sidebar expander
    with st.sidebar.expander("🐞 Debug Logs", expanded=False):
        for msg in st.session_state.debug_logs[-50:]:  # show last 50
            st.text(msg)


st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🔍 LangGraph-powered Chatbot with API Key Input")

# --- API Key Input ---
with st.sidebar:
    st.header("🔐 API Configuration")
    ak = st.text_input("Access key", type="password")
    sak = st.text_input("Secure access key", type="password")
    inf_prof = st.text_input("Inference profile", type="password")
    region = st.text_input("Region", type="password")
    use_keys = st.button("Use Keys")

# Store in session state when user submits
if use_keys:
    if ak:
        st.session_state["ACCESS_KEY"] = ak
        os.environ["AWS_ACCESS_KEY_ID"] = ak
    if sak:
        st.session_state["SECURE_ACCESS_KEY"] = sak
        os.environ["AWS_SECRET_ACCESS_KEY"] = sak
    if region:
        st.session_state["REGION"] = region
        os.environ["AWS_REGION"] = region
    if inf_prof:
        st.session_state["INFERENCE_PROFILE"] = inf_prof  # no need to set in os unless used
    st.success("API keys updated.")

# Ensure keys are available
if not st.session_state.get("ACCESS_KEY") or not st.session_state.get("SECURE_ACCESS_KEY"):
    st.warning("Please enter keys in the sidebar.")
    st.stop()

if "graph" not in st.session_state:
    st.write("🔧 Initializing LLM and Graph")  # DEBUG
    profile = st.session_state.get("INFERENCE_PROFILE", "")
    llm = ChatBedrock(model_id=profile + "us.amazon.nova-lite-v1:0", 
                      beta_use_converse_api=True,
                      provider="amazon", region_name=os.getenv("AWS_REGION","us-east-2"),
                      temperature=0.3)
    rag_graph   = build_rag_graph(llm)
    rag_tool    = Tool.from_function(rag_runner, name="RAG", description="RAG Tool")
    rag_tool_node = ToolNode(tools=[rag_tool])
    llm_tools   = llm.bind_tools([rag_tool])
    st.session_state.graph     = build_graph(llm_tools, rag_tool_node)
    st.session_state.thread_id = "1"
    st.session_state.messages  = []
else:
    st.write("✔️ Reusing existing graph")  # DEBUG

graph  = st.session_state.graph
thread = {"configurable": {"thread_id": st.session_state.thread_id}}
st.write("thread config:", thread)  # DEBUG

# ─── First-run bootstrap ────────────────────────────────────────────────────
if "snapshot" not in st.session_state:
    st.write("🚀 Bootstrapping to first interrupt")  # DEBUG
    for mode, snap in graph.stream({"messages": []}, thread, stream_mode=["values"]):
        st.write("boot_mode:", mode, "snap keys:", list(snap.keys()))  # DEBUG
        if "__interrupt__" in snap:
            st.write("💡 Got first interrupt:", snap["__interrupt__"])  # DEBUG
            st.session_state.snapshot = snap
            break

    # show prompt and stop
    intr_prompt = st.session_state.snapshot["__interrupt__"][0].value
    st.write("Showing interrupt prompt to user:", intr_prompt)  # DEBUG
    _ = st.chat_input(intr_prompt, key="resume_input")
    st.stop()

# ─── Render past history ─────────────────────────────────────────────────────
st.write("History:", st.session_state.messages)  # DEBUG
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Choose prompt text ──────────────────────────────────────────────────────
awaiting_input = "__interrupt__" in st.session_state.snapshot
st.write("awaiting_input =", awaiting_input)  # DEBUG

if awaiting_input:
    prompt_text = st.session_state.snapshot["__interrupt__"][0].value
    st.write("Prompting for interrupt response:", prompt_text)  # DEBUG
    user_text = st.chat_input(prompt_text, key="resume_input")
else:
    prompt_text = "Your message"
    st.write("Prompting for free-form chat:", prompt_text)  # DEBUG
    user_text = st.chat_input(prompt_text, key="new_message")

st.write("user_text:", repr(user_text))  # DEBUG

# ─── If user typed something ────────────────────────────────────────────────
if user_text:
    st.write("✅ Received user input:", user_text)  # DEBUG
    st.session_state.messages.append({"role": "user", "content": user_text})
    resume_cmd = Command(resume=user_text)
    st.write("Resume command:", resume_cmd)  # DEBUG

    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_full = ""
        st.write("▶️ Entering graph.stream")  # DEBUG

        for mode, payload in graph.stream(resume_cmd, thread, stream_mode=["messages", "values"]):
            st.write("stream mode:", mode, "payload keys:", list(payload.keys()))  # DEBUG

            if mode == "messages":
                chunk, _ = payload
                st.write("→ message chunk:", chunk)  # DEBUG
                text = (chunk.content if isinstance(chunk.content, str)
                        else "".join(seg["text"] for seg in chunk.content if seg.get("type")=="text"))
                assistant_full += text
                placeholder.markdown(assistant_full)

            elif mode == "values":
                st.write("→ values payload:", payload)  # DEBUG
                if "__interrupt__" in payload:
                    st.write("❗ New interrupt detected:", payload["__interrupt__"])  # DEBUG
                    st.session_state.snapshot = payload
                    st.write("Calling rerun() after interrupt")  # DEBUG
                    st.rerun()
                else:
                    st.write("✅ Final snapshot, finishing assistant turn")  # DEBUG
                    st.session_state.snapshot = payload
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_full}
                    )
                    st.write("Calling rerun() after completion")  # DEBUG
                    st.rerun()

# ─── Done ────────────────────────────────────────────────────────────────────
st.write("End of script reached")  # DEBUG
