from fastapi import FastAPI
from langserve import add_routes

from app.rag import build_rag_chain

# Crear app FastAPI
app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="RAG chatbot using LangChain + Ollama"
)

# Construir el chain RAG (el que ya funciona)
rag_chain = build_rag_chain()

# Exponer el chain como API
add_routes(
    app,
    rag_chain,
    path="/rag"
)

# Para ejecutar con: python server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
