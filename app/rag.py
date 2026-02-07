import os
import json
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data/promtior/data.pdf")
INDEX_PATH = os.path.join(BASE_DIR, "data/faiss_index")
INDEX_FILE = "index.pkl"
MODEL_META = os.path.join(INDEX_PATH, "model.json")

# Default model: tinyllama is used for low-RAM environments (e.g., Railway)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

def build_rag_chain():
    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Ensure the FAISS index directory exists
    os.makedirs(INDEX_PATH, exist_ok=True)
    index_path_full = os.path.join(INDEX_PATH, INDEX_FILE)

    # --- Check if index exists and matches the current model ---
    rebuild_index = True
    if os.path.exists(index_path_full) and os.path.exists(MODEL_META):
        with open(MODEL_META, "r") as f:
            meta = json.load(f)
        if meta.get("model") == OLLAMA_MODEL:
            rebuild_index = False  # Use existing index if model matches

    if rebuild_index:
        # Delete old index if it exists
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
            os.makedirs(INDEX_PATH, exist_ok=True)

        # Ensure the PDF exists
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"The PDF file was not found at {PDF_PATH}")

        # Load PDF and split into chunks
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        # Create FAISS vector store from chunks and embeddings
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)

        # Save metadata about the model used
        with open(MODEL_META, "w") as f:
            json.dump({"model": OLLAMA_MODEL}, f)
    else:
        # Load existing FAISS index
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Define a prompt template for RAG
    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the following context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}
""")

    # Ollama server URL (defaults to localhost)
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    # Build and return the RAG chain
    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
