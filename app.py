import streamlit as st
import os
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict

from graph_setup import build_rag_graph, build_graph, rag_runner

from langchain_core.tools import Tool
from langgraph.prebuilt.tool_node import ToolNode

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

# def langgraph_stream_to_text(graph, thread, initial_state):
#     state = initial_state
#     while True:
#         for mode, payload in graph.stream(
#                 state,
#                 thread,
#                 stream_mode=["messages", "values"]):
#             if mode == "messages":
#                 chunk, _ = payload
#                 # 1) If content is string
#                 if isinstance(chunk.content, str):
#                     yield chunk.content
#                 # 2) If content is list of segments
#                 else:
#                     for seg in chunk.content:
#                         if seg.get("type") == "text":
#                             yield seg["text"]
#             elif mode == "values" and "__interrupt__" in payload:
#                 # handle any interactive interrupts if needed…
#                 state = Command(resume=input("…"))  # adjust for Streamlit if desired
#                 break
#         else:
#             # normal completion
#             return

# if "messages" not in st.session_state:
#     st.session_state.messages = []
          
# thread = {"configurable": {"thread_id": "1"}}
# state = {"messages": []}

# st.title("LangGraph + Streamlit Chat")

# # User input
# if prompt := st.chat_input("Type your message"):
#     # 1. Record user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # 2. Stream assistant response
#     with st.chat_message("assistant"):
#         response = st.write_stream(
#             langgraph_stream_to_text(graph, thread, state)
#         )

#     # 3. Append final response to history
#     st.session_state.messages.append({"role": "assistant", "content": response})

def run_until_interrupt(state, thread):
    for mode, payload in graph.stream(state, thread, stream_mode=["messages","values"]):
        if mode == "messages":
            msg_chunk, _ = payload
            text = (msg_chunk.content 
                    if isinstance(msg_chunk.content, str) 
                    else "".join(seg["text"] for seg in msg_chunk.content if seg.get("type")=="text"))
            st.chat_message("assistant").write(text)
        elif mode == "values" and "__interrupt__" in payload:
            return state
    return state

# 5) Phase A: assistant turn
if not st.session_state.awaiting:
    st.info("🤖 Assistant is thinking…")
    st.session_state.state = run_until_interrupt(st.session_state.state, thread)
    st.session_state.awaiting = True

# 6) Phase B: user turn
if st.session_state.awaiting:
    user_input = st.chat_input("Your response:")  # only called once
    if user_input:
        st.chat_message("user").write(user_input)
        # Resume graph from interrupt
        graph.invoke(Command(resume=user_input), config=thread)
        st.session_state.awaiting = False
        # Immediately drain next assistant segment
        st.session_state.state = run_until_interrupt(st.session_state.state, thread)
        st.session_state.awaiting = True
