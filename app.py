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

if "state" not in st.session_state:
    st.session_state.state = {"messages": []}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper: inline user prompt for interrupts ---
def ask_user(prompt: str):
    return st.text_input(prompt, key=f"user_input_{len(st.session_state.messages)}")

# --- Render existing conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main user input box ---
user_text = st.chat_input("Your response:")
if user_text:
    # a) Append & render the new user message
    st.session_state.messages.append({"role": "user", "content": user_text})

    # b) Stream the assistant’s reply, handling interrupts inline
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        while True:
            for mode, payload in graph.stream(
                st.session_state.state,
                thread,
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    chunk, _ = payload
                    text = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else "".join(
                            seg["text"] 
                            for seg in chunk.content 
                            if seg.get("type") == "text"
                        )
                    )
                    full_reply += text
                    placeholder.markdown(full_reply)

                elif mode == "values":
                    # Interrupt: graph is waiting for more user input
                    if "__interrupt__" in payload:
                        intr = payload["__interrupt__"][0]
                        user_next = ask_user(intr.value)
                        if not user_next:
                            # pause until the user types something
                            return

                        # resume from the same point
                        st.session_state.state = Command(resume=user_next)
                        st.session_state.messages.append({"role": "user", "content": user_next})
                        break

                    # No more interrupts: finalise state & break both loops
                    else:
                        st.session_state.state = payload
                        placeholder.markdown(full_reply)
                        raise StopIteration
            else:
                # normal completion
                raise StopIteration

    # c) Save the assistant’s full reply in history
    st.session_state.messages.append({"role": "assistant", "content": full_reply})
