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

# Paths relativos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data/promtior/data.pdf")
INDEX_PATH = os.path.join(BASE_DIR, "data/faiss_index")
INDEX_FILE = "index.pkl"
MODEL_META = os.path.join(INDEX_PATH, "model.json")

# Modelo por defecto (cambia segun entorno)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")  # phi3 local, tinyllama Railway

def build_rag_chain():
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    os.makedirs(INDEX_PATH, exist_ok=True)
    index_path_full = os.path.join(INDEX_PATH, INDEX_FILE)

    # --- Revisar si el índice existe y si coincide el modelo ---
    rebuild_index = True
    if os.path.exists(index_path_full) and os.path.exists(MODEL_META):
        with open(MODEL_META, "r") as f:
            meta = json.load(f)
        if meta.get("model") == OLLAMA_MODEL:
            rebuild_index = False  # podemos usar el índice existente

    if rebuild_index:
        # Borra índice viejo si existía
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
            os.makedirs(INDEX_PATH, exist_ok=True)

        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"The PDF file was not found at {PDF_PATH}")

        # Cargar PDF y crear chunks
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)

        # Guardar metadata del modelo usado
        with open(MODEL_META, "w") as f:
            json.dump({"model": OLLAMA_MODEL}, f)
    else:
        # Cargar índice existente
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the following context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}
""")

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
