import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdfs(pdf_files):
    """Extract text from PDFs and split into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            loader = PyPDFLoader(tmp.name)
            pages = loader.load_and_split(text_splitter)
            all_chunks.extend(pages)
        os.unlink(tmp.name)  # Clean up temp file
    return all_chunks

def setup_retriever(chunks):
    """Create FAISS vector store and retriever."""
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

def main():
    st.title("📄 PDF RAG Chatbot")
    
    # Upload PDFs
    pdf_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if not pdf_files:
        st.info("Upload PDFs to begin.")
        return
    
    # Process PDFs and create retriever
    with st.spinner("Processing PDFs..."):
        chunks = process_pdfs(pdf_files)
        retriever = setup_retriever(chunks)
    
    # Initialize LLM (using Hugging Face Hub free tier)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )
    
    # Set up QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Chat interface
    st.subheader("Chat with your PDFs")
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})
            st.write("### Answer")
            st.write(result["result"])
            
            # Show sources
            st.write("### Source Documents")
            for doc in result["source_documents"]:
                st.write(f"- Page {doc.metadata['page']+1}: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()