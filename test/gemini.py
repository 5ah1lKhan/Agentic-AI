# import streamlit as st
# import pypdf
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# import os

import streamlit as st
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI  # Recommended import
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

# Set your OpenAI API key
# It's more secure to use st.secrets for this in a deployed app
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def process_pdfs(pdf_files):
    # This list will hold the document objects for all PDFs
    all_pages = []

    for pdf_file in pdf_files:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the PDF from the temporary path
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        all_pages.extend(pages)

        # Clean up the temporary file
        os.remove(tmp_file_path)

    if not all_pages:
        return None

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # The loader already gives us Document objects, so we can pass them directly
    texts = text_splitter.split_documents(all_pages)

    # Create embeddings and the vector store
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def main():
    st.title("Chat with Multiple PDFs")

    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Show a spinner while processing
        with st.spinner('Processing PDFs...'):
            vectorstore = process_pdfs(uploaded_files)

        if vectorstore:
            st.success("PDFs processed successfully!")

            # Create the RAG chain
            llm = OpenAI()
            # Use the vectorstore to create a retriever
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

            query = st.text_input("Ask a question about your documents:")

            if query:
                with st.spinner('Thinking...'):
                    response = qa_chain.run(query)
                st.write(response)
        else:
            st.error("Could not process any documents from the uploaded PDFs.")


if __name__ == '__main__':
    main()