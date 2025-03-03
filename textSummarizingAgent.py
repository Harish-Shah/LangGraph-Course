import os, getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from langgraph.graph import START, StateGraph, END

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("NVIDIA_API_KEY")

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

class State(MessagesState):
    legal_act: str  # Store the legal act text
    summary: str = ""  
    references: str = ""

# Node 1
def get_summary(state: State):
    content_to_summarize = state["legal_act"]
    summary_message = f"Summarize the following legal act:\n\n{content_to_summarize}"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    return {"summary": response.content, "messages": messages}

# Node 2
def get_references(state: State):
    references_message = f"List all Supreme Court judgments where the {state['summary']} was used in the judgment process."
    
    messages = state["messages"] + [HumanMessage(content=references_message)]
    response = model.invoke(messages)
    
    return {"references": response.content, "messages": messages}

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("get_summary", get_summary)
workflow.add_node("get_references", get_references)

workflow.add_edge(START, "get_summary")
workflow.add_edge("get_summary", "get_references")
workflow.add_edge("get_references", END)

graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# Run the chatbot workflow with a legal act
legal_act_text = "Name and territory of the Union 1. (1) India, that is Bharat, shall be a Union of States. [(2) The States and the territories thereof shall be as specified in the First Schedule]*. (3) The territory of India shall comprise- (a) the territories of the States; [(b) the Union territories specified in the First Schedule; and]** (c) such other territories as may be acquired. -------------------- * Subs. by the Constitution (Seventh Amendment) Act, 1956, s. 2, for cl. (2) ** Subs. by s. 2, ibid., for sub-clause (b)"
initial_state = State(messages=[], legal_act=legal_act_text)

output = graph.invoke(initial_state)
print("Summary:", output["summary"])
print("References:", output["references"])
