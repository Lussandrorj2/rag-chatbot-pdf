from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


DB_PATH = "vectorstore"

# 1️⃣ Carrega o banco vetorial
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True
    )

# 2️⃣ Busca documentos similares
def query_documents(query: str, k: int = 2):
    db = load_vectorstore()
    return db.similarity_search(query, k=k)

# 3️⃣ Inicializa o modelo local
llm = Ollama(
    model="phi3",
    temperature=0
)

# 4️⃣ Loop do chat
if __name__ == "__main__":
    print("🤖 Chat RAG iniciado (digite 'sair' para encerrar)\n")

    while True:
        # Pergunta do usuário
        user_query = input("Você: ")
        if user_query.lower() in ["sair", "exit", "quit"]:
            break

        # Busca no PDF
        docs = query_documents(user_query)

        # Cria o contexto (sem imprimir)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        # 🔥 PROMPT AQUI
        prompt = f"""
Você é um concierge de hotel.
Responda de forma clara, educada e objetiva.
Responda SOMENTE o que foi perguntado.
Não inclua informações extras.

Pergunta:
{user_query}

Contexto:
{contexto}

Resposta:
"""
        print("🤖 Pensando...")
        resposta = llm.invoke(prompt)

        print("\n🤖 Resposta:")
        print(resposta)

