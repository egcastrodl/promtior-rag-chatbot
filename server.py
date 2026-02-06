from fastapi import FastAPI
from langserve import add_routes

from app.rag import build_rag_chain

# Create FastAPI app
app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="RAG chatbot using LangChain + Ollama"
)

# Build the RAG chain
rag_chain = build_rag_chain()

# Expose the RAG chain as an API
add_routes(
    app,
    rag_chain,
    path="/rag"
)

# Run with: python server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
