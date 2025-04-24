import os, getpass
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Set NVIDIA API key
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("NVIDIA_API_KEY")

# Initialize AI Model
model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Request and Response Models
class SummarizeRequest(BaseModel):
    legal_act: str

class SummarizeResponse(BaseModel):
    summary: str

class ReferencesRequest(BaseModel):
    legal_act: str

class ReferencesResponse(BaseModel):
    references: str

# Node 1
def get_summary(legal_act: str) -> str:
    summary_prompt = f"Summarize the following legal act:\n\n{legal_act}"
    response = model.invoke([HumanMessage(content=summary_prompt)])
    return response.content

# Node 2
def get_references(legal_act: str) -> str:
    reference_prompt = (
        f"Find Supreme Court judgments where the following legal provision has been cited or referenced:\n\n{legal_act}."
        f" Provide case citations, case names, and a brief explanation of how the provision was applied."
    )
    response = model.invoke([HumanMessage(content=reference_prompt)])
    return response.content

# API Endpoints
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(data: SummarizeRequest):
    summary = get_summary(data.legal_act)
    return SummarizeResponse(summary=summary)

@app.post("/fetch-references", response_model=ReferencesResponse)
def fetch_references(data: ReferencesRequest):
    references = get_references(data.legal_act)
    return ReferencesResponse(references=references)

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Command to start server 
# uvicorn main:app --reload