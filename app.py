import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from rouge_score import rouge_scorer
import pandas as pd

from transformers import pipeline

generator = pipeline('text2text-generation', model='google/flan-t5-base')

def create_sample_medical_data(file_path="medical_data.txt"):
    sample_data = """
    Title: Early Symptoms of Alzheimer’s Disease
    Alzheimer’s disease is a progressive neurodegenerative disorder. Early symptoms include memory loss, difficulty concentrating, and challenges in planning or problem-solving. Patients may struggle with recalling recent events, recognizing familiar faces, or completing routine tasks. As the disease progresses, symptoms worsen, including confusion, mood swings, and language difficulties.

    Title: Treatment Options for Crohn's Disease
    Crohn’s disease is a chronic inflammatory bowel disease. Symptoms include abdominal pain, chronic diarrhea, weight loss, fatigue, and occasionally fever. Treatment options include anti-inflammatory drugs (e.g., corticosteroids), immunosuppressants, biologics, and lifestyle changes. In severe cases, surgery may be required to remove damaged portions of the intestine.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(sample_data)
    return True

def load_medical_data(file_path="medical_data.txt"):
    if not os.path.exists(file_path):
        create_sample_medical_data(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = DocArrayInMemorySearch.from_texts(texts=chunks, embedding=embedding_model)
    return vector_store

def generate_baseline_answer(question):
    prompt = f"Answer the question concisely: {question}"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text'].strip()

def rag_answer(vector_store, question, k=3):
    docs = vector_store.as_retriever(search_kwargs={"k": k}).get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Use the following medical context to answer the question accurately.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'].strip()

def evaluate_models(rag_answer, baseline_answer, question, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rag_rouge = scorer.score(reference, rag_answer)['rougeL'].fmeasure
    baseline_rouge = scorer.score(reference, baseline_answer)['rougeL'].fmeasure

    def calculate_f1(reference, prediction):
        ref_words = set(reference.lower().split())
        pred_words = set(prediction.lower().split())
        true_positives = len(ref_words & pred_words)
        precision = true_positives / len(pred_words) if pred_words else 0
        recall = true_positives / len(ref_words) if ref_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    rag_f1 = calculate_f1(reference, rag_answer)
    baseline_f1 = calculate_f1(reference, baseline_answer)

    return {
        "Question": question,
        "Baseline Answer": baseline_answer,
        "RAG Answer": rag_answer,
        "Baseline F1": baseline_f1,
        "RAG F1": rag_f1,
        "Baseline ROUGE-L": baseline_rouge,
        "RAG ROUGE-L": rag_rouge,
        "Notes": "RAG provides more context-based answers" if rag_rouge > baseline_rouge else "Baseline sufficient"
    }

def main():
    st.title("Medical Assistant with Flan-T5 (Open Source)")
    st.write("Ask a medical question, and the system will answer using a RAG approach and Flan-T5 baseline.")

    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vector_store = None

    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            create_sample_medical_data()
            chunks = load_medical_data()
            vector_store = create_vector_store(chunks)
            st.session_state.vector_store = vector_store
            st.session_state.initialized = True
            st.success("System initialized successfully!")

    query = st.text_input("Enter your medical question:", "What are the symptoms of Crohn's disease?")
    if st.button("Get Answer"):
        if st.session_state.initialized:
            with st.spinner("Generating answers..."):
                baseline = generate_baseline_answer(query)
                rag = rag_answer(st.session_state.vector_store, query)
                st.subheader("Answer (Baseline Flan-T5):")
                st.write(baseline)
                st.subheader("Answer (RAG Flan-T5):")
                st.write(rag)

                reference_answers = {
                    "What are the symptoms of Crohn's disease?": "Symptoms of Crohn's disease include abdominal pain, chronic diarrhea, weight loss, fatigue, and occasionally fever."
                }
                reference = reference_answers.get(query, "No reference answer available.")
                results = evaluate_models(rag, baseline, query, reference)
                if results:
                    st.subheader("Evaluation Results:")
                    df = pd.DataFrame([results])
                    st.markdown(df.to_markdown(index=False))
        else:
            st.error("System not initialized.")

if __name__ == "__main__":
    main()
