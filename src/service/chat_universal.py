import os, json, uuid
from typing import TypedDict, Annotated, AsyncIterator, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

MODEL = os.getenv("MODEL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
OPENAI_API_BASE = os.getenv("OPEN_AI_API_BASE", "")
 
CHAT_UNIVERSAL_SYSTEM_PROMPT = (
    "You are a concise, helpful assistant."
    "Answer clearly. If unsure, ask a short follow-up."
)


class ChatUniversalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def _make_graph():


    llm = ChatOpenAI(model=MODEL,
                     openai_api_key=OPENAI_API_KEY,
                     openai_api_base=OPENAI_API_BASE,
                     temperature=0)
    
    async def llm_node(state: ChatUniversalState) -> ChatUniversalState:
        resp = await llm.bind(stop=["<|eot_id|>"]).ainvoke(state["messages"])
        return {"messages": [resp]}
    
    general_graph_builder = StateGraph(ChatUniversalState)
    general_graph_builder.add_node("llm", llm_node)
    general_graph_builder.set_entry_point("llm")
    general_graph_builder.add_edge("llm", END)

    return general_graph_builder.compile(checkpointer=MemorySaver())

GRAPH = _make_graph()

# --- helpers ---
def _ensure_thread_id(thread_id: str | None) -> str:
    return thread_id or str(uuid.uuid4())

def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


# --- ChatService ---

class ChatUniversalService:
    def __init__(self):
        pass

    async def stream_chat(self, *, thread_id: str | None, message: str) -> AsyncIterator[str]:
        tid = _ensure_thread_id(thread_id)
        cfg = _config(tid)

        # prepend system (once) + new user message
        # NOTE: THIS code checks if there exists a system message inside the state['messages'] instead of checking if this is the start of the thread
        state = GRAPH.get_state(cfg)
        msgs: list[BaseMessage] = []
        existing = state.values.get("messages", []) if state and state.values else []
        if not any(isinstance(m, SystemMessage) for m in existing):
            msgs.append(SystemMessage(CHAT_UNIVERSAL_SYSTEM_PROMPT))
            # start event
            yield f"event: start\ndata: {json.dumps({'thread_id': tid})}\n\n"

        msgs.append(HumanMessage(message))

        # Stream model tokens out of the graph
        events = GRAPH.astream_events({"messages": msgs}, config=cfg, version="v2")
        async for event in events:
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # chunk.content is the token/partial text
                if getattr(chunk, "content", None):
                    yield f"event: chat_stream\ndata: {json.dumps({'content': chunk.content})}\n\n"

        # end event
        yield f"event: end\ndata: {json.dumps({'message': 'done'})}\n\n"