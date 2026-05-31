
# Bible-QA-RAG-Assistant
RAG-based Christian AI Assistant using FAISS, Hugging Face Embeddings, TinyLlama, and Gradio.
# BibleGuard AI
## 🚀 Live Demo

**Hugging Face Space:**  
https://huggingface.co/spaces/Ravishankarsharma/Bible-QA-Assistant
BibleGuard AI is a Retrieval-Augmented Generation (RAG) based Christian AI Assistant designed to provide scripture-grounded responses to Bible-related questions.

The application combines FAISS vector search, Hugging Face embeddings, and the TinyLlama language model to retrieve relevant biblical passages and generate contextual responses.

## Features

* Bible Question Answering
* Scripture Retrieval using FAISS
* Denomination Selection
* Prayer Generation
* Sermon Generation
* Devotional Generation
* Safety Filtering for Harmful Content
* Gradio Web Interface

## Technology Stack

* Python
* Gradio
* LangChain
* FAISS
* Hugging Face Transformers
* Sentence Transformers
* TinyLlama

## Project Architecture

User Query → FAISS Retrieval → Relevant Scripture Context → TinyLlama Generation → Final Response

## Installation

```bash
git clone https://github.com/yourusername/BibleGuard-AI.git
cd BibleGuard-AI

pip install -r requirements.txt
python app.py
```

## Deployment

The application can be deployed on Hugging Face Spaces using the Gradio SDK.

## Future Improvements

* Larger LLM Integration
* Bible Translation Support
* Citation Highlighting
* Multi-turn Conversation Memory

## Author

Ravi Shankar Sharma

Machine Learning Engineer | Bioinformatics | NLP | RAG Systems
