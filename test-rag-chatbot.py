from app.rag import build_rag_chain

chatbot = build_rag_chain()

print("Chatbot RAG listo (phi3 + PDF)\n")

while True:
    question = input("Me: ")
    response = chatbot.invoke(question)
    print("Bot:", response)
