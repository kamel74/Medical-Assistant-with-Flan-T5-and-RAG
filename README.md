# Medical Assistant with Flan-T5 and RAG

This project is a Streamlit-based web application that implements a medical question-answering system using the Flan-T5 model and Retrieval-Augmented Generation (RAG). It answers queries about medical conditions (e.g., Alzheimer’s and Crohn’s disease) by leveraging a vector store for context retrieval and compares baseline Flan-T5 answers with RAG-enhanced responses using ROUGE-L and F1 metrics.

## Features
- **Medical Q&A**: Users can input medical questions through a Streamlit web interface.
- **RAG Pipeline**: Retrieves relevant context from a vector store built with `medical_data.txt` using HuggingFace embeddings and DocArray.
- **Baseline vs. RAG**: Generates answers using Flan-T5 (`google/flan-t5-base`) with and without RAG, comparing their performance.
- **Evaluation Metrics**: Computes ROUGE-L and F1 scores to evaluate answer quality against reference answers.
- **Sample Data**: Includes `medical_data.txt` with information on Alzheimer’s and Crohn’s disease for context.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/medical-assistant-flan-t5-rag.git
   cd medical-assistant-flan-t5-rag
