from fastapi import FastAPI
from langserve import add_routes
from app.rag import build_rag_chain
import os

app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="RAG chatbot using LangChain + Ollama"
)

def get_rag_chain():
    return build_rag_chain()

add_routes(
    app,
    get_rag_chain,
    path="/rag"
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
