import streamlit as st
import pypdf # This import is not directly used but kept for context if you have other pypdf operations
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Import necessary class for Ollama
from langchain_community.llms import Ollama

import os
import tempfile

# No longer caching the LLM loading since Ollama handles the serving
# @st.cache_resource
def load_llm():
    """Loads the Ollama-hosted Mistral model."""
    # Ensure Ollama server is running and 'mistral' model is pulled
    llm = Ollama(model="mistral") # You can change "mistral" to any model you've pulled with Ollama
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

    # Make sure to install sentence-transformers: pip install sentence-transformers
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def main():
    st.title("Chat with Multiple PDFs using Local LLM (Ollama)")
    st.markdown("This demo uses **Ollama** to serve the LLM (e.g., Mistral-7B).")
    st.info("Make sure you have Ollama running and the desired model (e.g., `mistral`) pulled locally.")

    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner('Processing PDFs...'):
            vectorstore = process_pdfs(uploaded_files)

        if vectorstore:
            st.success("PDFs processed and indexed successfully!")
            with st.spinner('Connecting to Ollama and loading the Language Model...'):
                llm = load_llm() # This will connect to the Ollama server

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