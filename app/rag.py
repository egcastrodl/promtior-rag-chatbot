import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Rutas absolutas basadas en la ubicación de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data/promtior/data.pdf")
INDEX_PATH = os.path.join(BASE_DIR, "data/faiss_index")
INDEX_FILE = "index.pkl"  # Nombre del archivo FAISS

def build_rag_chain():
    # Embeddings para el vectorstore
    embeddings = OllamaEmbeddings(model="phi3")

    # Crear carpeta de índice si no existe
    os.makedirs(INDEX_PATH, exist_ok=True)

    # Path completo al archivo FAISS
    index_path_full = os.path.join(INDEX_PATH, INDEX_FILE)

    # Cargar índice si existe, si no crear desde PDF
    if os.path.exists(index_path_full):
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"El PDF no se encontró en {PDF_PATH}")

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)

    retriever = vectorstore.as_retriever()

    # Prompt para RAG
    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the following context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}
""")

    # URL de Ollama desde env var, default a localhost
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm = OllamaLLM(model="phi3", base_url=OLLAMA_URL)

    # Construir la cadena RAG
    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
