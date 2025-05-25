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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "state" not in st.session_state:
    st.session_state.state = {"messages": []}

if "awaiting" not in st.session_state:
    st.session_state.awaiting = False

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


INFERENCE_PROFILE_ARN = st.session_state["INFERENCE_PROFILE"] + "us.amazon.nova-lite-v1:0"

llm = ChatBedrock(
    model_id=INFERENCE_PROFILE_ARN,
    beta_use_converse_api=True,
    provider="amazon",
    region_name="us-east-2",
    temperature=0.3,
)
rag_graph = build_rag_graph(llm)
rag_tool = Tool.from_function(
    func=rag_runner,
    name="RAG",
    description="Performs RAG retrieval, reranking, and generation."
)
rag_tool_node = ToolNode(tools=[rag_tool])
llm_with_tools = llm.bind_tools([rag_tool])

if "graph" not in st.session_state:
    st.session_state["graph"] = build_graph(
        llm_with_tools, rag_tool_node
    )
graph = st.session_state["graph"] # since graph is stateful but rag_graph isn't
def run_until_interrupt(state, thread):
    """Yields ('assistant', text) or ('interrupt', full_text)."""
    collected = ""
    debug_log(f"▶ run_until_interrupt start, state={state!r}")
    for idx, (mode, payload) in enumerate(graph.stream(state, thread, stream_mode=["messages","values"])):
        debug_log(f"[{idx}] mode={mode!r}, payload={payload!r}")
        if mode == "messages":
            chunk, _ = payload
            text = (
                chunk.content
                if isinstance(chunk.content, str)
                else "".join(seg["text"] for seg in chunk.content if seg.get("type")=="text")
            )
            debug_log(f"   → chunk text: {text!r}")
            yield "assistant", text
            collected += text
        elif mode == "values" and "__interrupt__" in payload:
            debug_log("   → hit INTERRUPT")
            yield "interrupt", collected
            return
    debug_log("   → stream completed without interrupt")
    yield "complete", collected
    debug_log("▶ run_until_interrupt end")

# --- Main chat loop ---
thread = {"configurable": {"thread_id":"1"}}

# 1) Render existing history
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2) User turn
if user_text := st.chat_input("Your response:"):
    # a) Show user bubble
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role":"user","content":user_text}) # for rendering use
    debug_log(f"User input: {user_text!r}")

    # b) Resume the graph
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""
        for mode, payload in graph.stream(
                Command(resume=user_input),
                thread_config,
                stream_mode=["messages", "values"]
            ):
            if mode == "messages":
                chunk, _ = payload
                text = (
                    chunk.content
                    if isinstance(chunk.content, str)
                    else "".join(seg["text"] for seg in chunk.content if seg.get("type")=="text")
                )
                full_reply += text
                placeholder.markdown(full_reply)
            elif mode == "values" and "__interrupt__" in payload:
                # graph hit another interrupt
                break

    # c) Save assistant bubble
    st.session_state.messages.append({"role":"assistant","content":full_reply})
