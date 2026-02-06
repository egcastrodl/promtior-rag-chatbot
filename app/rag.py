from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

PDF_PATH = "data/promtior/data.pdf"
INDEX_PATH = "data/faiss_index"


def build_rag_chain():
    # Embeddings para el vectorstore
    embeddings = OllamaEmbeddings(model="phi3")

    # Cargar o crear índice FAISS
    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
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
    llm = Ollama(model="phi3", base_url=OLLAMA_URL)

    return (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )
