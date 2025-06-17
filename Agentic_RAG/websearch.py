from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
##for search engine
from langchain_community.tools import DuckDuckGoSearchResults
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

##load agent_template
def load_prompt_template(file_path = "/Users/sahilkhan/RAG/Agentic_RAG/template.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


##search engine
# def search(query: str) -> str:
#     duck = DuckDuckGoSearchResults(output_format="list")
#     result = duck.invoke(query)
#     return result.strip()

def search(query: str) -> str:
    """
    Executes a DuckDuckGo search and formats results like internal_document_retriever.
    """
    print(f"\n--- Calling web_search with query: '{query}' ---")
    duck = DuckDuckGoSearchResults(output_format="list")
    results = duck.invoke(query)

    if not results:
        return "No relevant search results found."

    retrieved_content = ""
    for i, item in enumerate(results):
        title = item.get("title", "No Title")
        snippet = item.get("snippet", "No Description")
        source_url = item.get("link", "unknown")

        retrieved_content += f"--- Result {i + 1} (Source: {source_url}) ---\n"
        retrieved_content += f"Title: {title}\n"
        retrieved_content += f"{snippet}\n\n"

    print(f"--- Retrieved {len(results)} search results. ---")
    return retrieved_content.strip()


#internal document search
def internal_document_retriever(query: str, retriever) -> str:
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

    agent_template = load_prompt_template()
    
    user_query = "What are Elliptic curve"
    # user_query = "What is the current population of India?" 
    # user_query = "Who won the IPL 2025"
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
    
        # --- Parsing the LLM's response ---
        json_match = re.search(r"```json\s*(\{.*?})\s*```", llm_response, re.DOTALL)
        end_of_action_delimiter_present = "<END_OF_ACTION>" in llm_response

        if "Final Answer:" in llm_response:
            # Before accepting Final Answer, ensure no Action-related keywords are present before it
            # This is a heuristic to catch cases where LLM tries to do both
            if json_match:
                error_msg = "CRITICAL ERROR: LLM provided both an Action and a Final Answer. You must choose one or the other. Please correct your response."
                print(f"Agent Error: {error_msg}")
                agent_scratchpad += f"\nObservation: {error_msg}"
                continue # Skip to next turn, providing feedback


            final_answer = llm_response.split("Final Answer:", 1)[1].strip()
            print(f"\n--- Agent provided Final Answer: ---")
            print(final_answer)
            break

        elif json_match and end_of_action_delimiter_present:
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
                if not isinstance(action_dict, dict):
                    error_msg = f"LLM's Action JSON is not a dictionary. Expected format: {{'tool_name': '...', 'tool_input': '...'}}."
                    raise ValueError(error_msg)

                # Check for required explicit keys
                if "tool_name" not in action_dict:
                    error_msg = "LLM's Action JSON is missing the 'tool_name' key. Expected format: {'tool_name': '...', 'tool_input': '...'}"
                    raise ValueError(error_msg)
                if "tool_input" not in action_dict:
                    error_msg = "LLM's Action JSON is missing the 'tool_input' key. Expected format: {'tool_name': '...', 'tool_input': '...'}"
                    raise ValueError(error_msg)
                
                # If there are extra keys beyond 'tool_name' and 'tool_input', it's also an error
                if len(action_dict) != 2:
                    error_msg = f"LLM's Action JSON should contain exactly 'tool_name' and 'tool_input' keys. Found {len(action_dict)} keys."
                    raise ValueError(error_msg)


                tool_name = action_dict['tool_name']
                tool_input = action_dict['tool_input']

                # Validate tool_name (keep existing check)
                if tool_name not in ["internal_document_retriever", "search"]:
                    error_msg = f"LLM specified an unknown tool '{tool_name}'. Available tools are 'internal_document_retriever' and 'search'."
                    raise ValueError(error_msg)
                
                # Check if tool_input is a string (keep existing check)
                if not isinstance(tool_input, str):
                    error_msg = f"Tool input for '{tool_name}' must be a string. Received type: {type(tool_input)}."
                    raise ValueError(error_msg)
                

                print(f"Parsed Tool Name: {tool_name}")
                print(f"Parsed Tool Input: {tool_input}")

                # Append Action to scratchpad
                agent_scratchpad += f"\nAction:\n```json\n{json_str}\n```"

                # Execute the tool
                observation = ""
                if tool_name == "internal_document_retriever":
                    observation = internal_document_retriever(tool_input, retriever)
                elif tool_name == "search":
                    observation = search(tool_input)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'. Available tools: internal_document_retriever."
                # print(f"Observation:{observation}")
                
                # Append Observation to scratchpad
                agent_scratchpad += f"\nObservation: {observation}"
                print(f"Observation added to scratchpad. Scratchpad length: {len(agent_scratchpad)}")

            except json.JSONDecodeError as e:
                error_msg = f"LLM failed to provide valid JSON. Error: {e}. Ensure JSON is correctly formatted like {{'tool_name': '...', 'tool_input': '...'}}."
                print(f"Agent Error: {error_msg}")
                agent_scratchpad += f"\nObservation: {error_msg}"
            except ValueError as e: # Catch our custom ValueErrors for format issues
                error_msg = f"LLM provided an invalid Action format: {e}. Please ensure the JSON has explicit 'tool_name' and 'tool_input' keys with string values."
                print(f"Agent Error: {error_msg}")
                agent_scratchpad += f"\nObservation: {error_msg}"
            except Exception as e:
                error_msg = f"An unexpected error occurred during tool execution: {e}. Please ensure the tool input is correct and relevant to the tool."
                print(f"Agent Error: {error_msg}")
                agent_scratchpad += f"\nObservation: {error_msg}"
            
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