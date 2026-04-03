from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_query

app = FastAPI(title = "Financial Risk RAG API")

class QueryRequest(BaseModel):
  query:str

@app.get("/")
def health_check():
  return {"status":"ok"}

@app.post("/query")
def query_rag(request:QueryRequest):
  answer = ask_query(request.query)
  return {"query" : request.query, "answer" : answer} 
