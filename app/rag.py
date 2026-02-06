import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Use paths relative to repo root
PDF_PATH = os.path.join(os.getcwd(), "data/promtior/data.pdf")
INDEX_PATH = os.path.join(os.getcwd(), "data/faiss_index")
INDEX_FILE = "index.pkl"

def build_rag_chain():
    embeddings = OllamaEmbeddings(model="phi3")

    # Ensure FAISS directory exists
    os.makedirs(INDEX_PATH, exist_ok=True)

    index_path_full = os.path.join(INDEX_PATH, INDEX_FILE)

    # Load existing FAISS index or create new one
    if os.path.exists(index_path_full):
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"The PDF file was not found at {PDF_PATH}")

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

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the following context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}
""")

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm = OllamaLLM(model="phi3", base_url=OLLAMA_URL)

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
