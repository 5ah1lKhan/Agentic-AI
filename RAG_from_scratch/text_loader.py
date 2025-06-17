from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains import SelfAskWithSearchChain
from langchain.utilities import SerpAPIWrapper


import os

"""
EXPERIMENT

loader1 = PyPDFLoader("./RAG_from_scratch/sample.pdf")
loader1 = PyMuPDFLoader("./RAG_from_scratch/sample.pdf")
loader1 = PDFMinerLoader("./RAG_from_scratch/sample.pdf")

docs = loader1.load()
docs = loader1.load_and_split()
docs = loader1.lazy_load()

print(docs)

pages=[]
for page in loader1.lazy_load():
    pages.append(page)
print(pages)
"""

def process_pdfs(pdf_files):
    """Processes uploaded PDF files and returns a vector store."""
    all_pages = []
    for pdf_file in pdf_files:
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        #     tmp_file.write(pdf_file.getvalue())
        #     tmp_file_path = tmp_file.name
        try:
            from langchain_community.document_loaders import PyPDFLoader
            # loader = PyPDFLoader(tmp_file_path)
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            all_pages.extend(pages)
        finally:
            # os.remove(tmp_file_path)
            print("loaded successfully")


    if not all_pages:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(all_pages)
    # print(len(texts))
    # print(texts[0])

    # Make sure to install sentence-transformers: pip install sentence-transformers
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectorstore = FAISS.from_documents(texts, embeddings)
    # return vectorstore
    return texts

text = process_pdfs(["./RAG_from_scratch/sample.pdf"])

# Import necessary classes
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load an Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create Embeddings and a Vector Store
# This step automatically processes your 'chunks' using the 'embeddings' model
vectorstore = FAISS.from_documents(text, embeddings)

print("Vector store created successfully!")

# Optional: Perform a simple similarity search to test
# query = "Regional Trends"
# docs = vectorstore.similarity_search(query)

# print("\n--- Retrieved Documents for Query: '{}' ---".format(query))
# for doc in docs:
#     print("start from here")
#     print(doc.page_content)
#     print("-" * 30) # Separator for readability

# llm = Ollama(model="mistral")

# print("\n--- LLM (Mistral via Ollama) initialized successfully! ---")

# You'll need to set your Hugging Face API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


##conversational mode, so require InferenceClient
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Or any other public model
#     temperature=0.7,
#     max_new_tokens=1024
# )

##this is working
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", # Or any other public model
    temperature=0.7,
    max_new_tokens=1024
)
# llm = HuggingFaceHub(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     model_kwargs={"max_new_tokens": 512, "temperature": 0.7}
# )


print("Hugging face model activates")

# Optional: Test the LLM with a simple query
# This is just to confirm the connection, not part of the RAG chain yet
# test_query = "What is the capital of France?"
# try:
#     response = llm.invoke(test_query)
#     print(f"Test LLM response to '{test_query}':")
#     print(response)
# except Exception as e:
#     print(f"Error connecting to Ollama or generating response: {e}")
#     print("Please ensure Ollama is running and the 'mistral' model is pulled.")

retriever = vectorstore.as_retriever()
print("Retriever created from vector store.")

##single question - single answer
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff", # Other common types: "map_reduce", "refine", "map_rerank"
#     retriever=retriever,
#     return_source_documents=True # Set to True to see which documents were retrieved
# )

# print("RetrievalQA chain created.")

# # 3. Formulate a Query and Run the Chain
# query = "What are the sources of energy?" # Make this relevant to your sample_document.txt

# print(f"\n--- Asking the RAG system: '{query}' ---")

# # Use .invoke() for newer LangChain versions
# result = qa_chain.invoke({"query": query})

# # The result object contains the answer and (if return_source_documents=True) the retrieved docs
# answer = result["result"]
# source_documents = result["source_documents"]

# print("\n--- Answer from RAG System ---")
# print(answer)

# print("\n--- Source Documents Used ---")
# for i, doc in enumerate(source_documents):
#     print(f"Document {i+1} (Page Content):")
#     print(doc.page_content)
#     print("-" * 30) # Separator

search = SerpAPIWrapper()
self_ask_chain = SelfAskWithSearchChain(llm=llm,retriever=retriever )

self_ask_chain.run("What is the capital of the country whose leader is Justin Trudeau?")