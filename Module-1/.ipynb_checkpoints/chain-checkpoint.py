import os, getpass
from pprint import pprint
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import AIMessage, HumanMessage

def _set_env(var: str):
    if not os.environ.get(var):
        # os.environ[var] = getpass.getpass(f"{var}: ")
        os.environ[var] = "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"
        
_set_env("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

# tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])

# print(tool_call)

# State
class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass


# Node
def tool_calling_llm(state: MessagesState):
    print(state["messages"])
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

#Invoking
#The LLM chooses to use a tool when it determines that the input or task requires the functionality provided by that tool.
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})

for m in messages['messages']:
    m.pretty_print()