import os, getpass
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("NVIDIA_API_KEY")

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

class State(MessagesState):
    legal_act: str
    summary: str = ""  
    references: str = ""

# Node 1
def get_summary(state: State):
    content_to_summarize = state.get("legal_act", "")
    summary_message = f"Summarize the following legal act:\n\n{content_to_summarize}"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    return {"summary": response.content, "messages": messages}

# Node 2
def get_references(state: State):
    content_to_get_references = state.get("summary", "")
    
    references_message = "List all Supreme Court judgments where this was used in the judgment process." + content_to_get_references
    
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

memory= MemorySaver()
graph = workflow.compile(interrupt_after=["get_summary"], checkpointer=memory)

legal_text = "THE CONSTITUTION (FIFTH AMENDMENT) ACT, 1955 [24th December, 1955.] An Act further to amend the Constitution of India.  BE it enacted by Parliament in the Sixth Year of the Republic of India as follows:---  1. Short title.-This Act may be called the Constitution (Fifth Amendment) Act, 1955.  2. Amendment of article 3.-In article 3 of the Constitution, for the proviso, the following proviso shall be substituted, namely:-  Provided that no Bill for the purpose shall be introduced in either House of Parliament except on the recommendation of the President and unless, where the proposal contained in the Bill affects the area, boundaries or name of any of the States specified in Part A or Part B of the First Schedule, the Bill has been referred by the President to the Legislature of that State for expressing its views thereon within such period as may be specified in the reference or within such further period as the President may allow and the period so specified or allowed has expired."

initial_state = State(messages=[], legal_act=legal_text)
thread = {"configurable": {"thread_id": "1"}}

output = graph.invoke(initial_state, thread)
print("Summary:", output["summary"])
print("-----"*25)

# Get user feedback
user_approval = input("Do you want to fetch related references as well? (yes/no): ")

if user_approval.lower() == "yes":
    
    output = graph.invoke(None, thread)
    print("References:", output["references"])
        
else:
    print("Operation cancelled by user.")
    