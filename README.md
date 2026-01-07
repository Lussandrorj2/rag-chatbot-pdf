# 📄 RAG Chatbot com Embeddings Locais

Projeto de **Recuperação Aumentada por Geração (RAG)** para consulta semântica em documentos PDF, utilizando embeddings locais e FAISS, sem dependência de APIs externas.

## 🧠 Tecnologias
- Python
- LangChain
- HuggingFace Embeddings
- FAISS
- PyPDF

## 🏗️ Arquitetura
PDFs → Chunking → Embeddings → FAISS → Busca Semântica

## ▶️ Como rodar o projeto

```bash
# Criar ambiente virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Coloque seus PDFs em /data

# Criar base vetorial
python ingest.py

# Consultar documentos
python chat.py
