from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
import json
import re

# from langchain.tools import Tool
# from langchain.agents import AgentExecutor, create_react_agent, initialize_agent
# from langchain.agents.agent_types import AgentType
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Used for the agent's specific prompt


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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)

    # Make sure to install sentence-transformers: pip install sentence-transformers
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    # embeddings = HuggingFaceEmbeddings(model_name="/Users/sahilkhan/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2",model_kwargs={"device": device})
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully!")
    return vectorstore

stop_sequences = [
    "Observation:",
    "Final Answer:",
    "\n\n",
    "<|user|>"
]

def load_llm():
#     llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta", # Or any other public model
#     temperature=0.01,
#     max_new_tokens=300,
#     stop_sequences=stop_sequences,
#     repetition_penalty=1.1
# )
    # llm = Ollama(model="mistral")
    llm = Ollama(model="phi3",temperature = 0)
    print("LLM activates")
    return llm

def document_retriever(query: str, retriever) -> str:
    """
    Executes the document retrieval based on the provided query.
    Returns the page content of the top relevant documents as a single string.
    """
    print(f"\n--- Calling document_retriever with query: '{query}' ---")
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    # Concatenate the content of the retrieved documents
    # Add source information if available in metadata
    retrieved_content = ""
    for i, doc in enumerate(docs):
        source_info = doc.metadata.get("source", "unknown")
        retrieved_content += f"--- Document {i+1} (Source: {source_info}) ---\n"
        retrieved_content += doc.page_content + "\n\n"
    print(f"--- Retrieved {len(docs)} documents. ---")
    return retrieved_content.strip()

def main():

    sample_pdf = ["/Users/sahilkhan/RAG/Agentic_RAG/sample.pdf"]
    vectorstore = process_pdfs(sample_pdf)

    retriever = vectorstore.as_retriever()
    print("Retriever created from vector store.")

    llm = load_llm()

    agent_template = """You are an intelligent assistant. Your goal is to answer the user's query *only* using information available in your agent_scratchpad(which contains prior Thoughts, Actions, and Observations). If the agent_scratchpad does not contain enough information, use the provided tools.s
You have access to the following tool:

**document_retriever**:
  - Description: Use this tool to find factual information within the documents.
  - Input: A concise, relevant string query for searching the documents (e.g., "What is RAG?").
  - Returns: Relevant chunks of text from the documents.

Your response MUST STRICTLY follow one of these two formats:

Option 1: Perform an Action
When you need to use a tool to gather more information:
Thought: Your reasoning for needing more information and which tool you will use.
Action:
```json
{{"tool_name": "tool_input"}}
```
(Replace "tool_name" with "document_retriever" and "tool_input" with your precise query.)
<END_OF_ACTION>

Option 2: Provide a Final Answer
When you have sufficient information from the Observations to answer the user's query:
Thought: Your final reasoning process, reviewing the observations.
Final Answer: [Your complete and accurate answer based only on the Observations]

CRITICAL RULES:

You MUST include a "Thought:" in every response.
You MUST provide EITHER an Action (JSON block followed by <END_OF_ACTION>) OR a Final Answer (preceded by a Thought:). NEVER BOTH.
Do NOT generate any speculative Observation: text. You will not receive any Observations from the tool execution.Your task ends here without getting the observation.
Do NOT provide a Final Answer unless you have used the document_retriever tool to obtain the necessary factual information from the documents if the query requires it.
The user's query is: {input}

Begin!

agent_scratchpad :  {agent_scratchpad}"""

    
    user_query = "What are Elliptic curve"
    agent_scratchpad = "" # Initialize as empty string
    max_iterations = 5 # Safety limit for the loop
    final_answer = None

    for i in range(max_iterations):
        print(f"\n--- Agent Turn {i+1} ---")
        current_prompt = agent_template.format(input=user_query, agent_scratchpad=agent_scratchpad)

        # Send the current prompt to the LLM
        try:
            llm_response = llm.invoke(current_prompt)
            print(f"LLM Raw Response:\n{llm_response}")
        except Exception as e:
            print(f"Error during LLM invocation in turn {i+1}: {e}")
            break
        # print("llm response end here")
        # print("\n")
        # --- Parsing the LLM's response ---
        json_match = re.search(r"```json\s*(\{.*?})\s*```", llm_response, re.DOTALL)
        end_of_action_delimiter_present = "<END_OF_ACTION>" in llm_response

        if json_match and end_of_action_delimiter_present:
            json_str = json_match.group(1)
            # Check for any text after the <END_OF_ACTION> delimiter
            remaining_text_after_delimiter = llm_response.split("<END_OF_ACTION>", 1)[1].strip()
            if remaining_text_after_delimiter:
                error_msg = f"LLM generated extra text after <END_OF_ACTION> delimiter. Response: {llm_response}"
                print(error_msg)
                agent_scratchpad += f"\nObservation: {error_msg}. You must STOP immediately after the Action block and <END_OF_ACTION>."
                continue # Go to next turn, LLM needs to correct itself

            try:
                action_dict = json.loads(json_str)
                # print(f"json str: {json_str}")
                # tool = list(action_dict.keys())[0]
                tool_name = action_dict['tool_name']
                tool_input = action_dict['tool_input']

                print(f"Parsed Tool Name: {tool_name}")
                print(f"Parsed Tool Input: {tool_input}")

                # Append Action to scratchpad
                agent_scratchpad += f"\nAction:\n```json\n{json_str}\n```"

                # Execute the tool
                observation = ""
                if tool_name == "document_retriever":
                    observation = document_retriever(tool_input, retriever)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'. Available tools: document_retriever."
                # print(f"Observation:{observation}")
                
                # Append Observation to scratchpad
                agent_scratchpad += f"\nObservation: {observation}"
                print(f"Observation added to scratchpad. Scratchpad length: {len(agent_scratchpad)}")

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing JSON from LLM response: {e}. LLM response was:\n{llm_response}"
                print(error_msg)
                agent_scratchpad += f"\nObservation: {error_msg}. LLM failed to provide valid JSON. Agent must retry."
            except Exception as e:
                error_msg = f"Error during tool execution: {e}"
                print(error_msg)
                agent_scratchpad += f"\nObservation: {error_msg}. Agent must retry."
        
        elif "Final Answer:" in llm_response:
            # Before accepting Final Answer, ensure no Action-related keywords are present before it
            # This is a heuristic to catch cases where LLM tries to do both
            if "Action:" in llm_response.split("Final Answer:", 1)[0] or "```json" in llm_response.split("Final Answer:", 1)[0]:
                error_msg = "LLM attempted to provide both an Action and a Final Answer. You must provide EITHER an Action OR a Final Answer. Please correct your response."
                print(f"Agent Error: {error_msg}")
                agent_scratchpad += f"\nObservation: {error_msg}"
                continue

            final_answer = llm_response.split("Final Answer:", 1)[1].strip()
            print(f"\n--- Agent provided Final Answer: ---")
            print(final_answer)
            break
        
        else:
            # If neither valid Action nor Final Answer, it's a format violation
            error_msg = "LLM did not provide a valid Action (JSON + <END_OF_ACTION>) or Final Answer. Please re-evaluate your reasoning and follow the format strictly."
            print(f"Agent Error: {error_msg}")
            agent_scratchpad += f"\nObservation: {error_msg}" # Provide feedback to LLM
            # Also, ensure Thought is captured if present, before adding error observation
            thought_match = re.search(r"Thought:\s*(.*?)(?=(?:Action:)|(?:Final Answer:)|$)", llm_response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""
            if thought and not agent_scratchpad.endswith(f"Thought: {thought}"): # Avoid duplicate thoughts
                agent_scratchpad += f"\nThought: {thought}"

    if final_answer is None:
        print("\n--- Agent could not find a Final Answer within max iterations. ---")
        print("Last scratchpad content:")
        print(agent_scratchpad)


if __name__ == '__main__':
    main()