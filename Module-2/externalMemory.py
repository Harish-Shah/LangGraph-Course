# Chatbot with message summarization
import os, getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from IPython.display import Image, display
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# In memory
conn = sqlite3.connect(":memory:", check_same_thread = False)

db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        
_set_env("NVIDIA_API_KEY")

_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state : State):
    
    # Get summary if it exists
    summary = state.get("summary", "")
    
    if summary :
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
        
    else:
        messages = state["messages"]
        
    response = model.invoke(messages)
    return {"messages" : response}
    

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")
    
     # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# Adding Memory

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation",should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))

# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
# input_message = HumanMessage(content="hi! I'm Haris")
# output = graph.invoke({"messages": [input_message]}, config) 
# for m in output['messages'][-1:]:
#     m.pretty_print()

# input_message = HumanMessage(content="what's my name?")
# output = graph.invoke({"messages": [input_message]}, config) 
# for m in output['messages'][-1:]:
#     m.pretty_print()

# input_message = HumanMessage(content="i like the Indian Cricket Team!")
# output = graph.invoke({"messages": [input_message]}, config) 
# for m in output['messages'][-1:]:
#     m.pretty_print()

    
print(graph.get_state(config).values.get("summary",""))