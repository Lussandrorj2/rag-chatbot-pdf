from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = "vectorstore"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

def query_documents(query: str, k: int = 3):
    db = load_vectorstore()
    results = db.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    print("🤖 Chat RAG iniciado (digite 'sair' para encerrar)\n")

    while True:
        user_query = input("Você: ")
        if user_query.lower() in ["sair", "exit", "quit"]:
            break

        docs = query_documents(user_query)

        print("\n📄 Trechos mais relevantes encontrados:\n")
        for i, doc in enumerate(docs, start=1):
            print(f"[{i}] {doc.page_content[:500]}...\n")
