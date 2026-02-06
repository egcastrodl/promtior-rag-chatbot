# Project Overview

## Objective
The goal of this challenge was to develop a chatbot assistant that answers questions about Promtior using a RAG (Retrieval-Augmented Generation) architecture with LangChain.

## Implemented Solution
1. **Document Loading**: Used `PyPDFLoader` to load the `data.pdf` as the source of information.  
2. **Text Splitting**: Applied `RecursiveCharacterTextSplitter` to split the content into manageable chunks, with overlap to preserve context.  
3. **Vectorization and Storage**: Used `OllamaEmbeddings` to convert the chunks into vectors, stored locally with FAISS for retrieval.  
4. **LLM Integration**: Integrated `Ollama` (model `phi3`) as the local LLM for generating responses.  
5. **RAG Chain**: Built a RAG chain combining the retriever and the LLM to answer questions based on the PDF content.  

## Key Features
- Answers questions specifically using the content of `data.pdf`.  
- Returns clear and concise answers using the local Ollama model.  
- Supports offline execution without requiring cloud LLM services.  

## Challenges and Solutions
- **Memory constraints**: Initially used a larger model which exceeded available RAM. Switched to `phi3` for compatibility with limited resources.  
- **Deprecation warnings**: Updated code to use `langchain-ollama` where necessary.  
- **Local vs Cloud deployment**: Chose to run everything locally to avoid cloud costs, making it simpler and free to test.

## Next Steps / Deployment
- Can be deployed on Railway or other cloud services. Using Ollama locally allows free execution without cloud GPU costs.  
- The solution is ready for demonstration without further cloud setup.
