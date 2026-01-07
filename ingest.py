import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_PATH = "vectorstore"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

if __name__ == "__main__":
    print("📄 Lendo PDFs...")
    documents = load_documents()

    print("✂️ Quebrando documentos em chunks...")
    chunks = split_documents(documents)

    print("🧠 Criando embeddings locais e salvando no FAISS...")
    create_vectorstore(chunks)

    print("✅ Ingestão finalizada com sucesso!")
