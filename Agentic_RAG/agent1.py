from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import os

from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Used for the agent's specific prompt


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def process_pdfs(pdf_files):
    """Processes uploaded PDF files and returns a vector store."""
    all_pages = []
    for pdf_file in pdf_files:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            all_pages.extend(pages)
        finally:
            print("loaded successfully")


    if not all_pages:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(all_pages)

    # Make sure to install sentence-transformers: pip install sentence-transformers
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully!")
    return vectorstore

def load_llm():
    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", # Or any other public model
    temperature=0.7,
    max_new_tokens=1024
)
    print("LLM activates")
    return llm

def main():

    sample_pdf = ["/Users/sahilkhan/RAG/Agentic_RAG/sample.pdf"]
    vectorstore = process_pdfs(sample_pdf)
    retriever = vectorstore.as_retriever()
    print("Retriever created from vector store.")
    llm = load_llm()

    ##tools

    document_retriever_tool = Tool(
        name = "document_retriver",
        func=lambda query: retriever.get_relevant_documents(query),
        description="Useful for answering questions about the content of the provided documents. Input should be a question or keyword phrase to search for relevant information."
    )
    tools = [document_retriever_tool]
    tool_names = ["document_retriver"]
    print(f"Tools available to the agent: {[tool.name for tool in tools]}")

    # prompt = ChatPromptTemplate.from_template([
    #     ("system", "You are a helpful assistant. You have access to the following tools:\n\n{tools}\n\n, tool_names:\n\n{tool_names}\n\n, Use them to answer user questions."),
    #     ("human", "{input}"),
    #     ("placeholder", "{agent_scratchpad}"), # This is where the agent's thoughts and tool interactions go
    #     ])
    # prompt = ChatPromptTemplate.from_template([
    #     ("system", "You are a helpful assistant. You have access to the tools,Use them to answer user questions."),
    #     ("human", "{input}"),
    #     ("placeholder", "{agent_scratchpad}"), # This is where the agent's thoughts and tool interactions go
    #     ])
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant. You have access to the following tools:\n\n{tools}\n\n, tool_names:\n\n{tool_names}\n\n, Use them to answer user questions."),
    #     ("human", "{input}"),
    #     MessagesPlaceholder("agent_scratchpad"),
    # ])
    # print("Agent prompt template defined.")

    # agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    # print("Agent created.")

    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # print("Agent Executor created.")    

    # query = "What are the key components involved in learning about RAG systems? Explain step by step." # Make this a question that requires tool use

    # print(f"\n--- Asking the Agentic RAG system: '{query}' ---")

    # try:
    #     result = agent_executor.invoke({"input": query}) # Use 'input' as the key for the prompt
    #     print("\n--- Final Answer from Agentic RAG System ---")
    #     print(result["output"]) # The agent executor returns a dictionary with 'output' key
    # except Exception as e:
    #     print(f"An error occurred while running the agent: {e}")
    #     print("Please ensure Ollama is running and the 'mistral' model is pulled.")

    # Initialize the structured chat agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    print("Agent created successfully.")

    query = "What are the key components involved in learning about RAG systems? Explain step by step."

    print(f"\n--- Asking the Agentic RAG system: '{query}' ---")

    try:
        result = agent.invoke({"input": query})
        print("\n--- Final Answer from Agentic RAG System ---")
        print(result)
    except Exception as e:
        print(f"An error occurred while running the agent: {e}")



if __name__ == '__main__':
    main()

