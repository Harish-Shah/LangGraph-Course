import os, getpass
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.utilities import SQLDatabase

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("NVIDIA_API_KEY")

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

db = SQLDatabase.from_uri("postgresql://anc:admin@localhost:5432/gid_db")

class State(MessagesState):
    user_query : str
    sql_query : str
    sql_query_result : str
    readle_resp : str
    
# Node 1
def generate_sql_query(state : State):
    
    return

# Node 2
def execute_sql_query(state : State):
    
    return

# Node 3
def generate_readable_resp(state : State):
    
    return

# defining workflow
workflow = StateGraph(State)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_sql_query", execute_sql_query)
workflow.add_node("generate_readable_resp", generate_readable_resp)

workflow.add_edge(START, "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_sql_query")
workflow.add_edge("execute_sql_query", "generate_readable_resp")
workflow.add_edge("generate_readable_resp", END)

graph = workflow.compile()

user_query = ""

initial_state = State(messages=[], user_query=user_query)
output = graph.invoke(initial_state)