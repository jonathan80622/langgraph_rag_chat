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
    profile = st.session_state.get("INFERENCE_PROFILE", "")
    llm = ChatBedrock(
        model_id=profile + "us.amazon.nova-lite-v1:0",
        beta_use_converse_api=True,
        provider="amazon",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        temperature=0.3,
    )

    rag_graph   = build_rag_graph(llm)
    rag_tool    = Tool.from_function(rag_runner, name="RAG",
                                     description="Retrieval-augmented answer")
    rag_tool_node = ToolNode(tools=[rag_tool])
    llm_tools   = llm.bind_tools([rag_tool])

    st.session_state.graph     = build_graph(llm_tools, rag_tool_node)
    st.session_state.thread_id = "1"   # fixed thread for this demo
    st.session_state.messages  = []    # visible chat history

graph  = st.session_state.graph
thread = {"configurable": {"thread_id": st.session_state.thread_id}}

# ─── First-run bootstrap: drive graph until first interrupt & checkpoint ────
if "snapshot" not in st.session_state:
    for _mode, snap in graph.stream({"messages": []}, thread,
                                    stream_mode=["values"]):
        if "__interrupt__" in snap:
            st.session_state.snapshot = snap    # full StateSnapshot
            break
    # render prompt box and stop – user must respond next run
    intr_prompt = st.session_state.snapshot["__interrupt__"][0].value
    st.chat_input(intr_prompt, key="resume_input")
    st.stop()

# ─── Helper: render past conversation bubbles ───────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Decide which prompt to show: interrupt vs. normal chat ────────────────
awaiting_input = "__interrupt__" in st.session_state.snapshot

if awaiting_input:
    prompt_text = st.session_state.snapshot["__interrupt__"][0].value
    user_text   = st.chat_input(prompt_text, key="resume_input")
else:
    prompt_text = "Your message"
    user_text   = st.chat_input(prompt_text, key="new_message")

# ─── If user just typed something, resume or start the graph run ────────────
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    resume_cmd = Command(resume=user_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_full = ""

        for mode, payload in graph.stream(resume_cmd, thread,
                                          stream_mode=["messages", "values"]):
            if mode == "messages":
                chunk, _ = payload
                if isinstance(chunk.content, str):
                    assistant_full += chunk.content
                else:
                    assistant_full += "".join(
                        seg["text"] for seg in chunk.content
                        if seg.get("type") == "text"
                    )
                placeholder.markdown(assistant_full)

            elif mode == "values":
                # ❗ NEW interrupt → save snapshot & rerun
                if "__interrupt__" in payload:
                    st.session_state.snapshot = payload
                    st.experimental_rerun()
                # ✅ Finished assistant turn
                else:
                    st.session_state.snapshot = payload
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_full}
                    )
                    st.experimental_rerun()
