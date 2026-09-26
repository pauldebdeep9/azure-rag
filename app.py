from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="Sai Satcharitra RAG")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Sai Satcharitra RAG API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from excss