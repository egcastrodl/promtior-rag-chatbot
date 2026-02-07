from fastapi import FastAPI
from langserve import add_routes
from app.rag import build_rag_chain
import os

app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="RAG chatbot using LangChain + Ollama"
)

# Build RAG chain
rag_chain = build_rag_chain()

# Expose the RAG chain
add_routes(
    app,
    rag_chain,
    path="/rag"
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
