import streamlit as st
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Import necessary classes for Hugging Face
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch  # PyTorch is required for transformers

import os
import tempfile

# Caching the model loading for better performance
@st.cache_resource
def load_llm():
    """Loads a quantized Mistral model and tokenizer."""
    # Model name for a conversational, quantized Mistral model
    model_name = "HuggingFaceH4/zephyr-7b-beta"

    # Load the tokenizer and a quantized version of the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for reduced memory usage
        device_map="auto",        # Automatically use GPU if available
        trust_remote_code=True,
    )

    # Create a text-generation pipeline
    # Note: For CausalLM models, the task is "text-generation"
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # The maximum number of tokens to generate
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def process_pdfs(pdf_files):
    """Processes uploaded PDF files and returns a vector store."""
    all_pages = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            all_pages.extend(pages)
        finally:
            os.remove(tmp_file_path)

    if not all_pages:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_pages)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def main():
    st.title("Chat with Multiple PDFs using Open-Source LLMs")
    st.markdown("This demo uses **Mistral-7B** from Hugging Face.")

    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner('Processing PDFs...'):
            vectorstore = process_pdfs(uploaded_files)

        if vectorstore:
            st.success("PDFs processed and indexed successfully!")
            with st.spinner('Loading the Language Model... (This can take a few minutes on first run)'):
                llm = load_llm()

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            query = st.text_input("Ask a question about the content of your documents:")
            if query:
                with st.spinner('Generating answer...'):
                    response = qa_chain.run(query)
                st.write(response)
        else:
            st.error("Could not process the uploaded PDF files.")

if __name__ == '__main__':
    main()