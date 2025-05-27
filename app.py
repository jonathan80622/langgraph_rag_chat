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

# ————— Initialize or reuse the LangGraph “chatbot” ————————————————————
# ————— Initialize or reuse the LangGraph “chatbot” ————————————————————
if "graph" not in st.session_state:
    st.write("🔧 Initializing LangGraph and Bedrock client")  # DEBUG
    profile = st.session_state.get("INFERENCE_PROFILE", "")
    llm = ChatBedrock(
        model_id=profile + "us.amazon.nova-lite-v1:0",
        beta_use_converse_api=True,
        provider="amazon",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        temperature=0.3,
    )

    # Build RAG + chat graph
    rag_graph     = build_rag_graph(llm)
    rag_tool      = Tool.from_function(rag_runner, name="RAG",
                                       description="Retrieval-augmented answer")
    rag_tool_node = ToolNode(tools=[rag_tool])
    llm_with_tools= llm.bind_tools([rag_tool])
    st.session_state.graph     = build_graph(llm_with_tools, rag_tool_node)
    st.session_state.thread_id = "1"
    st.session_state.messages  = []
    st.write("✅ Graph initialized and saved")  # DEBUG

graph  = st.session_state.graph
thread = {"configurable": {"thread_id": st.session_state.thread_id}}
st.write("📌 Using thread ID:", thread)  # DEBUG

# ————— Step 1: Bootstrap to the first interrupt ——————————————————————
if "snapshot" not in st.session_state:
    st.write("🚀 First run — bootstrapping graph to first interrupt")  # DEBUG
    for mode, snap in graph.stream({"messages": []}, thread, stream_mode=["values"]):
        st.write("→ stream mode:", mode)  # DEBUG
        st.write("→ snapshot keys:", list(snap.keys()))  # DEBUG
        if "__interrupt__" in snap:
            st.session_state.snapshot = snap
            st.write("💡 Got first interrupt:", snap["__interrupt__"])  # DEBUG
            st.write("st.session_state.snapshot is now", st.session_state.snapshot)
            st.write("st.session_state.snapshot is now", st.session_state.snapshot)
            
            break
    else:
        st.error("❌ Graph returned no interrupt on startup.")
        st.stop()

prompt = st.session_state.snapshot["__interrupt__"][0].value
st.write("🛑 Prompting user with interrupt:", prompt)  # DEBUG
resume_input = st.chat_input(prompt)

if resume_input:
    
    # ————— Step 2: Show chat history ————————————————————————————————
    st.write("📜 Rendering conversation history:")  # DEBUG
    for msg in st.session_state.messages:
        st.write(f"{msg['role'].upper()}: {msg['content']}")  # DEBUG
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # ————— Step 3: Handle next user input ———————————————————————————————
    st.write("snapshot")
    st.write(f"snapshot looks like {st.session_state.snapshot}")
    st.write("SNAPSHOT")
    prompt = st.session_state.snapshot["__interrupt__"][0].value
    st.write("🔁 Waiting for user input — prompt:", prompt)  # DEBUG
    user_input = st.chat_input(prompt, key="resume_input")
    st.write("✍️ User typed:", user_input)  # DEBUG
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        cmd = Command(resume=user_input)
        st.write("🧠 Resuming graph with Command:", cmd)  # DEBUG
    
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            st.write("▶️ Streaming from LangGraph...")  # DEBUG
    
            for mode, payload in graph.stream(cmd, thread, stream_mode=["messages", "values"]):
                st.write("📡 stream mode:", mode)  # DEBUG
    
                if mode == "messages":
                    chunk, _ = payload
                    st.write("💬 message chunk:", chunk)  # DEBUG
                    text = (chunk.content if isinstance(chunk.content, str)
                            else "".join(seg["text"] for seg in chunk.content
                                         if seg.get("type") == "text"))
                    full += text
                    placeholder.markdown(full)
    
                elif mode == "values":
                    st.write("📦 values payload keys:", list(payload.keys()))  # DEBUG
                    if "__interrupt__" in payload:
                        st.session_state.snapshot = payload
                        st.write("🔂 New interrupt received, looping...")  # DEBUG
                        st.rerun()
                    else:
                        st.session_state.snapshot = payload
                        st.session_state.messages.append(
                            {"role": "assistant", "content": full}
                        )
                        st.write("✅ Assistant reply complete, looping...")  # DEBUG
                        st.rerun()
    
    # ————— END ————————————————————————————————————————————————
    st.write("✅ Script completed; waiting for user input.")  # DEBUG
