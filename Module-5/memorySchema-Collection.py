import os, getpass
from pydantic import BaseModel, Field
from trustcall import create_extractor
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def _set_env(var: str):
    if not os.environ.get(var):
        print(f"Set {os.environ[var]} in environment variables.")
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("NVIDIA_API_KEY")

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")

trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True,
)

# Instruction
instruction = """Extract memories from the following conversation:"""

# Conversation
conversation = [HumanMessage(content="Hi, I'm Haris."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="This morning I had a nice session of reading.")]

# Invoke the extractor
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=instruction)] + conversation})

# Messages contain the tool calls
for m in result["messages"]:
    m.pretty_print()