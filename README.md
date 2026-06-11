<img width="1620" height="1050" alt="image" src="https://github.com/user-attachments/assets/6663a3b2-c7e4-402e-b842-ae2d6a15f242" />

### An AI Lens for ardent readers and librarians alike — powered entirely by open-source AI models.

----


## 🔦 OVERVIEW

**BookHead** is an AI-driven application that **helps book readers to find their next read** and **librarians to organize and classify book shelves and generate comprehensive book catalogs**, just by clicking a picture of book shelves. Built on a stack of **open-source models** like Gemma-2B LLM, PaddleOCR and Faster RCNN. It uses highly specialised book detection, OCR, book embeddings, cataloging, and recommendation pipelines — useful for personal libraries, archiving projects, and research collections.

## ⚡Features

1. **Smart book detection** and **OCR** from uploaded shelf images
2. **Embeddings-based semantic search** and **recommendations**
3. **Automated cataloging** and **metadata extraction** (title, author, ISBN)
4. **Configurable confidence routing** and **review queue** for manual verification
5. **Highly Scalable architecture** with quick background workers for extensive tasks
6. Pluggable **LLM** and **vector store** integrations for flexible retrieval

---

## **🧩 TECH STACK**

| ✨ Category             | 🤖 Tools / Libraries              |
| -------------------- | ------------------------------ |
| **Language**         | Python 3.11+                    |
| **Web Framework**    | FastAPI + Uvicorn               |
| **Background processing**    | Celery (with Redis broker)               |
| **Vector store**    | Qdrant (or configurable alternative)             |
| **Storage**    | local uploads (persistant) / cloud-capable (S3-compatible)               |
| **ML/AI**    | 1. Book Detection pipeline <br> 2. OCR pipeline <br> 3. Embedding models <br> 4. Light-weight insystem LLM + configurable API alternative   |
| **AI Models**    | 1. Faster RCNN <br> 2. PaddleOCR (Text Detection + Text Recognition) <br> 3. all-MiniLM-L6-V2 <br> 4. Gemma-2B LLM + option to plugin LLM API|
| **Containerization**    | Docker + docker-compose          |
| **Deployment**       | Azure Cloud - VM           |
| **Container Registry**    | Azure Container Registry          |
| **Version Control**  | Git + GitHub          |


---


