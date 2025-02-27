import os, getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

def _set_env(var: str):
    if not os.environ.get(var):
        # os.environ[var] = getpass.getpass(f"{var}: ")
        os.environ[var] = "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"
        
_set_env("NVIDIA_API_KEY")

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

#Node

def assistant(state: MessagesState):
    return {"messages" : [llm_with_tools.invoke([sys_msg] + state["messages"])]}

#Graph

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools",ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
# adding new edge to return back to the assitant node from the ToolNode.
builder.add_edge("tools", "assistant")

react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2 and then Divide the output by 5")]

messages = react_graph.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()