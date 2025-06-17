from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
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
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully!")
    return vectorstore

stop_sequences = [
    "Observation:",
    "Final Answer:",
    "\n\n",
    "<|user|>",
]

def load_llm():
#     llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta", # Or any other public model
#     temperature=0.01,
#     max_new_tokens=300,
#     stop_sequences=stop_sequences,
#     repetition_penalty=1.1
# )
    llm = Ollama(model="mistral")
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

    # agent_template = """Answer the following questions as best you can. You have access to the following tools:

    #             **document_retriever**: Useful for answering questions about the content of the provided documents. Input to this tool should be a string query (e.g., "what is RAG?"). This tool will return relevant document chunks.

    #             The way you use the tools is by specifying a json blob.
    #             Specifically, you must call a tool by using a json blob. The json blob must have a single key-value pair, where the key is the name of the tool and the value is the input to the tool.
    #             Provide your reasoning in a "Thought:" section before performing any Action.

    #             For example:
    #             Thought: I need to find information about X. I will use the document_retriever tool.
    #             ```json
    #             {{"document_retriever": "X"}}
    #             Observation: [Result of the tool]

    #             If you have found the answer to the question, state your final answer in the format:
    #             Final Answer: [Your answer here]

    #             The user's query is: {input}

    #             Begin!

    #             {agent_scratchpad}"""

#     agent_template = """You are an intelligent assistant designed to answer questions based *only* on the provided documents.
# You have access to the following tool:

# **document_retriever**:
#   - Description: Useful for finding factual information within the documents.
#   - Input: A concise string query relevant to the information you need.
#   - Returns: Relevant chunks of text from the documents.

# Follow these steps strictly:
# 1. **Think**: Always begin with a "Thought:" explaining your reasoning, what information you need, and which tool you plan to use (if any).
# 2. **Act**: If you need to use a tool, output a JSON action block. The JSON MUST have a single key-value pair where the key is the tool name ("document_retriever") and the value is the tool input (your query for the tool).
#    Example Action:
#    ```json
#    {{"document_retriever": "query to search"}}
# Observe: After the tool runs, you will receive an "Observation:" containing the tool's output. Incorporate this into your next "Thought:".
# Final Answer: Once you have gathered enough information from the documents to answer the user's question completely and accurately, state your final answer clearly in the format: "Final Answer: [Your answer here]". Do NOT provide a Final Answer without using the document_retriever if the question requires factual information from the documents.
# The user's query is: {input}

# Begin!

# {agent_scratchpad}"""

    # agent_template = """You are an intelligent assistant that on given a user_input and agent_scratchpad decides whether to provide answers based on the information in agent_scratchpad or take a Action.
    #     You have access to the following tool to take Action:

    #     **document_retriever**:
    #     - Description: Use this tool to find factual information within the documents.
    #     - Input: A concise, relevant string query for searching the documents.
    #     - Returns: Relevant chunks of text from the documents.

    #     Your process MUST follow this cycle:
    #     1.if you have Observation in agent_scratchpad : {agent_scratchpad} Use this information in your Thought.
    #     2. **Thought**: Explain your reasoning for the current step, what information you need, and which tool you will use.
    #     3. Always include **Thought** in your response
    #     4. Now after **Thought** You can either choose **Action** or **Final answer**
    #     5. **Action**: If you need to use a tool, output a JSON action block. The JSON MUST have a single key-value pair.
    #     Example Action (ONLY use this format for actions):
    #     ```json
    #     {{"document_retriever": "your precise search query for the documents"}}
    #     6.STOP immediately after the JSON block if you are performing an Action. DO NOT write anything else and return the response so far, and leave the rest on me. Thank you for the response
    #     7. **Final Answer**: if in your response **Action** is not included then Once you have gathered sufficient information from the Observations to accurately and completely answer the user's question, provide your final answer in this exact format: "Final Answer: [Your answer here]".
    #     DO NOT provide a Final Answer if you have not used the document_retriever tool to obtain the necessary factual information from the documents.

    #     CRITICAL : in your response there can either be **Action** or **Final Answer**, If you hallucinate and add lines such as "(After receiving the response from the document_retriever)" and try to include both in your response, it is a negative response. Please try to avoid it

    #     The user's query is: {input}

    #     Begin!
    #     """

    agent_template = """You are an intelligent assistant that provides answers *ONLY* based on the information retrieved from the provided documents.
You have access to the following tool:

**document_retriever**:
  - Description: Use this tool to find factual information within the documents.
  - Input: A concise, relevant string query for searching the documents (e.g., "What is RAG?").
  - Returns: Relevant chunks of text from the documents.

Your response MUST STRICTLY follow one of these two formats:

**Option 1: Perform an Action**
When you need to use a tool to gather more information:
Thought: Your reasoning for needing more information and which tool you will use.
```json
{{"tool_name": "tool_input"}}
(Replace "tool_name" with "document_retriever" and "tool_input" with your precise query.)
STOP IMMEDIATELY after the JSON block. Do NOT write anything else.

Option 2: Provide a Final Answer
When you have sufficient information from the Observations to answer the user's query:
Thought: Your final reasoning process, reviewing the observations.
Final Answer: [Your complete and accurate answer based only on the Observations]

CRITICAL RULES:

You MUST include a "Thought:" in every response.
You MUST provide EITHER an Action (JSON block followed by STOP) OR a Final Answer (preceded by a Thought:). NEVER BOTH.
Do NOT generate any speculative Observation: text. You will receive Observations from the tool execution.
Do NOT provide a Final Answer unless you have used the document_retriever tool to obtain the necessary factual information from the documents if the query requires it.
The user's query is: {input}

Begin!

{agent_scratchpad}"""





    
    user_query = "What is RAG"
    agent_scratchpad = "Observation: " ""
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
        # 1. Check for Final Answer
        if "Final Answer:" in llm_response:
            final_answer = llm_response.split("Final Answer:", 1)[1].strip()
            print(f"\n--- Agent provided Final Answer: ---")
            print(final_answer)
            break

        # 2. Extract Thought
        # thought_match = re.search(r"Thought:\s*(.*?)(?=(?:```json)|(?:Final Answer:)|$)", llm_response, re.DOTALL)
        thought_match = re.search(r"Thought:\s*(.*?)(?=(?:Action:)|(?:Final Answer:)|$)", llm_response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if thought:
            agent_scratchpad += f"\nThought: {thought}"

        # 3. Extract Action and Action Input (JSON)
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                action_dict = json.loads(json_str)
                tool_name = list(action_dict.keys())[0]
                tool_input = action_dict[tool_name]

                print(f"Parsed Tool Name: {tool_name}")
                print(f"Parsed Tool Input: {tool_input}")

                # Append Action to scratchpad
                # agent_scratchpad += f"\nAction: {tool_name}\nAction Input: {tool_input}"
                agent_scratchpad += f"\nAction:\n```json\n{json_str}\n```" # Append the exact JSON it generated


                # Execute the tool
                observation = ""
                if tool_name == "document_retriever":
                    observation = document_retriever(tool_input, retriever)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'."
                
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
        else:
            # If no JSON found, but no final answer, the LLM is probably just thinking
            # We need to tell it it didn't call a tool if it should have.
            # This is where LLM coaxing is important. If it just outputs thought, it's fine.
            # If it was supposed to call a tool, we need to guide it.
            print("No tool call (JSON) found in LLM response for this turn. LLM might be thinking or off-track.")
            if "Thought:" in llm_response and not json_match and not "Final Answer:" in llm_response:
                # If it's just a thought, let it continue. No new observation to add regarding tool.
                pass
            else:
                # This could be a case where LLM didn't follow format properly
                error_msg = "LLM did not provide a valid tool call or final answer. Please re-evaluate your reasoning and follow the tool call format."
                print(f"Agent Error: {error_msg}")
                agent_scratchpad += f"\nObservation: {error_msg}" # Provide feedback to LLM

    if final_answer is None:
        print("\n--- Agent could not find a Final Answer within max iterations. ---")
        print("Last scratchpad content:")
        print(agent_scratchpad)


    # initial_prompt = agent_template.format(input=user_query, agent_scratchpad="")

    # print("\n--- LLM's First Response (raw) ---")
    # try:
    #     llm_response = llm.invoke(initial_prompt)
    #     print(llm_response)
    # except Exception as e:
    #     print(f"Error during initial LLM invocation: {e}")
    #     print("Please ensure your Hugging Face API token is valid and the model is accessible.")




if __name__ == '__main__':
    main()

