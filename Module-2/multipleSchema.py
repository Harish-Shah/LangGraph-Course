from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

# -> PrivateState indicated the state we writing out to. OverallState here indicated the schema of the node
def node_1(state : OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz" : state['foo'] + 1}

def node_2(state : PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo" : state['baz'] + 1}

# Build graph
builder = StateGraph(OverallState)
builder.add_node("node_1",node_1)
builder.add_node("node_2",node_2)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Add
graph = builder.compile()

# View
# display(Image(graph.get_graph().draw_mermaid_png()))

# print(graph.invoke({"foo" : 5}))

## Input / Output Schema

# By default, `StateGraph` takes in a single schema and all nodes are expected to communicate with that schema.
# However, it is also possible to define explicit input and output schemas for a graph.

class InputState(TypedDict):
    question : str

class OutputState(TypedDict):
    answer : str
    
class OverallState(TypedDict):
    question: str
    answer: str
    notes: str
    
def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his name is Haris"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Haris"}

graph = StateGraph(OverallState, input=InputState, output=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

print(graph.invoke({"question":"hi"}))