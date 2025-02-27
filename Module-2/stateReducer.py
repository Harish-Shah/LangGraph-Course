from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from operator import add
from typing import Annotated

class State(TypedDict):
    foo : Annotated[list[int], add]
    
def node_1(state):
    print("---Node 1---",state)
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state):
    print("---Node 2---",state)
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    print("---Node 3---",state)
    return {"foo": [state['foo'][-1] + 1]}

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()

print(graph.invoke({"foo" : [10]}))