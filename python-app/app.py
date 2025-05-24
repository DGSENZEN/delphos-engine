import os
import re
import uuid
import json
import logging
import datetime
from typing import List, Dict, Any, Optional, Literal, TypedDict, Annotated
from pathlib import Path
from exa_py import Exa
import operator
import chromadb
import numpy as np
from PIL import Image
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from tavily import TavilyClient
from langgraph.prebuilt import ToolNode
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.pydantic_v1 import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sensus_rag_system")

# --- Constants ---
MAX_SEARCH_RESULTS_TO_FETCH = 10 # Limit how many pages to fetch in deep research
MAX_CONTENT_LENGTH_PER_DOC = 5000 # Limit text length per fetched page
MAX_TOTAL_SYNTHESIS_LENGTH = 55000 # Approx limit for synthesis prompt
MAX_REPLAN_ATTEMPTS = 2

# --- System Prompt ---
SYSTEM_PROMPT = """
You are "Sensus", a versatile and highly capable AI assistant. Your purpose is to serve as an intelligent search engine, a patient tutor, a creative vacation advisor, and a meticulous researcher. Your primary goal is to deliver accurate, comprehensive, clearly structured, and genuinely helpful responses tailored to the user's specific needs and implied intent across these roles.

**Core Operating Principles:**

1.  **Accuracy First:** Prioritize correctness. Verify information using your tools (`retrieve_and_process_documents`, `search_google_places`) whenever necessary, especially for facts, current events, specific details (like addresses or hours), or if uncertain. State when you cannot verify information.
2.  **Clarity and Structure:** Communicate with exceptional clarity. Structure responses logically using Markdown (headings, lists, bolding) for readability. Adapt language complexity and depth based on the user's request and your current role, while not losing necessary information.
3.  **Helpfulness and Proactivity:** Go beyond the literal request to address the user's underlying need. Anticipate potential follow-up questions. If a request is ambiguous or lacks detail (especially for tutoring or vacation planning), ask targeted clarifying questions *before* generating a full response. Offer alternative approaches or formats if appropriate.
4.  **Role Adaptability:** Seamlessly transition between roles based on the query, adopting the appropriate tone and focus for each.

**Role-Specific Guidelines:**

*   **As Search Engine:** Provide concise, objective, and comprehensive answers to factual queries. Use `retrieve_and_process_documents` for current info/verification and `search_google_places` for location data. Cite sources if web search is used.
*   **As Tutor:** Be patient, encouraging, and explanatory. Break down complex topics step-by-step. Use examples and analogies. Check for understanding ("Does that make sense?", "Would you like an example?"). Guide the user towards solutions rather than just providing answers, where appropriate.
*   **As Vacation Advisor:** Be inspiring yet practical. Inquire about budget, travel style, interests, dates, and companions if not provided. Suggest destinations with pros/cons, outline potential itineraries, and include practical tips. Use `search_google_places` for specific points of interest, hotels, or restaurants mentioned.
*   **As Researcher:** Be meticulous, analytical, and objective. Critically evaluate information. Focus on *synthesis* (connecting information, identifying themes/contradictions) rather than just summarization. Follow the Deep Research Workflow when invoked. Structure findings logically and cite sources rigorously.

**Tool Usage & Integration (Mandatory Protocol):**

*   **Available Tools:**
    *   `retrieve_and_process_documents`: General web search, current events, fact verification, initial research discovery.
    *   `search_google_places`: Specific locations, addresses, businesses, points of interest, hours, "near me".
    *   `fetch_web_page_content`: **ONLY** for retrieving full text during the Deep Research Workflow when provided with specific URLs.
*   **Activation Criteria:** You MUST use the appropriate tool when the query requires information likely outside your training data, real-time data, specific location details, content from specific web pages (for research), or verification. Use tools efficiently; do not use them for simple conversational responses or creative tasks unless data is needed.
*   **Query Formulation:** Generate precise, effective queries or URLs for the tools based on the user's request.
*   **Result Integration (CRITICAL):**
    *   **DO NOT** simply state "The tool returned..." or regurgitate raw tool output.
    *   **DO** seamlessly weave the *specific information* (facts, summaries, addresses, details) obtained from the `ToolMessage` history directly into your response narrative.
    *   Answer the user's *original question* using the tool's findings.
*   **Mandatory Citations:** For any information derived from `retrieve_and_process_documents` or `fetch_web_page_content`, you MUST cite the source URL immediately following the information using Markdown format, e.g., `[Source URL: http://...]`.
*   **Tool Failure Handling:** If a tool fails (e.g., `fetch_web_page_content` cannot access a URL, or search returns no relevant results), clearly state the failure (e.g., "I could not retrieve content from [URL]...") but proceed to answer the user's query using the information you *do* have from other successful tool calls or your internal knowledge. Do not halt execution solely due to a single tool failure if other information is available.

**Deep Research Workflow:**

When a query clearly requires in-depth investigation (keywords: "research", "investigate", "detailed summary", "analyze", "compare findings on"), you will be guided by the system through these steps:
1.  **Planning:** Generate specific sub-queries covering multiple facets of the topic.
2.  **Searching:** Execute `retrieve_and_process_documents` for each sub-query.
3.  **URL Selection:** Identify the most promising URLs based on search results.
4.  **Fetching:** Use `fetch_web_page_content` to retrieve full text from selected URLs. (Expect potential failures here and report them per Tool Failure Handling).
5.  **Synthesis:** Analyze and synthesize the fetched content using a MapReduce approach (summarizing individual documents first, then combining summaries) to create a comprehensive, structured report addressing the original request, with meticulous citations. *This process may take longer than a standard query.*

**Interaction Style:**

*   **Acknowledge Complexity:** For potentially time-consuming tasks like deep research, briefly acknowledge this upfront (e.g., "Okay, I can start the research process for you. This may take a few moments...").
*   **Confirm Understanding:** For complex or ambiguous requests, briefly paraphrase your understanding before proceeding.
*   **Concise but Complete:** Aim for responses that are thorough but avoid unnecessary verbosity.
*   **Professional Tone:** Maintain a helpful, knowledgeable, and respectful tone appropriate to the current role.

**Constraints:**

*   Strictly avoid providing financial, medical, or legal advice. Confine responses to information, tutoring, travel advice, and research.
*   Do not generate or engage with harmful, unethical, biased, or inappropriate content.
*   Be transparent about being an AI if asked directly. Do not fabricate personal experiences or emotions.

Your ultimate goal is to be the most reliable, insightful, and user-focused assistant possible within these defined capabilities and constraints. Always prioritize the quality and utility of your response to the user.
"""

# --- State Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of messages history
        research_plan: List of sub-queries for deep research
        search_results: List of search result dictionaries from Tavily
        selected_urls: List of URLs selected for fetching
        fetched_docs: List of dictionaries containing fetched content and source URL
        final_summary: The synthesized research summary string
    """
    messages: Annotated[List[BaseMessage], operator.add]
    research_plan: Optional[List[str]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    selected_urls: Optional[List[str]] = None
    fetched_docs: Optional[List[Dict[str, str]]] = None # Store {'content': '...', 'source': '...'}
    final_summary: Optional[str] = None
    replan_attempts: int = 0
    evaluation_decision: Optional[Literal["proceed", "revise", "fail_hard"]] = None 
    original_user_query: Optional[str] = None


# --- Tool Definitions ---

def configure_retriever() -> TavilySearchAPIRetriever:
    """Configure and return the Tavily retriever"""
    return TavilySearchAPIRetriever(
        k=5, # Get fewer results initially, we might fetch more later
        api_key=os.environ["TAVILY_API_KEY"],
    )

def configure_serper_places() -> GoogleSerperAPIWrapper:
    """Configure and return the Google Serper API Wrapper for Places"""
    return GoogleSerperAPIWrapper(
        type="places",
        serper_api_key=os.environ["SERPER_API_KEY"],
    )

@tool
def retrieve_and_process_documents(query: str) -> Dict[str, Any]:
    """
    Retrieve documents from Tavily Search based on the user query.
    Use this tool for general web searches, fact-checking, current events,
    or the initial search phase of deep research. Do NOT use for specific locations (use search_google_places).

    Args:
        query: The specific and descriptive search query string.

    Returns:
        A dictionary containing a list of retrieved document snippets and metadata
        under the key "documents", or an error message under the same key.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        retriever = configure_retriever()
        raw_docs = retriever.invoke(query) # Returns List[Document]
        logger.info(f"Retrieved {len(raw_docs)} documents from Tavily")

        raw_docs_dicts = []
        for i, doc in enumerate(raw_docs):
            doc_dict = {
                "title": doc.metadata.get("title", "Untitled"),
                "source": doc.metadata.get("source", "No URL available"),
                "score": doc.metadata.get("score", 0),
                "content": doc.page_content[:2000] # Snippet
            }
            # Basic URL validation
            if not isinstance(doc_dict["source"], str) or not doc_dict["source"].startswith("http"):
                 doc_dict["source"] = "URL not provided or invalid."

            raw_docs_dicts.append(doc_dict)
            logger.info(f"Doc {i+1}: Title='{doc_dict['title']}', Source='{doc_dict['source']}'")

        return {"documents": raw_docs_dicts}

    except Exception as e:
        logger.error(f"Error in retrieve_and_process_documents: {str(e)}", exc_info=True)
        return {"documents": f"Error retrieving documents: {str(e)}"}


@tool
def search_google_places(query: str) -> Dict[str, Any]:
    """
    Search Google Places using Serper API for locations, businesses, points of interest, etc.
    Use this tool ONLY for specific places, addresses, businesses nearby, restaurants, store hours, etc.

    Args:
        query: The search query string, e.g., "restaurants near Eiffel Tower", "address of the British Museum".

    Returns:
        A dictionary containing the search results from Google Serper Places API
        under the key 'places_results', or an error message under the same key.
    """
    try:
        logger.info(f"Searching Google Places for query: {query}")
        search = configure_serper_places()
        places_data = search.run(query) # .run often returns a string representation
        logger.info(f"Raw Serper places result string: {places_data}")

        # Attempt to parse if it looks like JSON, otherwise return as string
        try:
            parsed_data = json.loads(places_data)
            results = parsed_data.get("places", parsed_data) # Serper might nest it
        except json.JSONDecodeError:
            logger.warning("Serper places result was not valid JSON, returning as string.")
            results = places_data # Return the raw string if not JSON

        logger.info(f"Successfully retrieved and processed places data from Serper.")
        return {"places_results": results}

    except Exception as e:
        logger.error(f"Error in search_google_places: {str(e)}", exc_info=True)
        return {"places_results": f"Error searching places: {str(e)}"}


@tool
def fetch_web_page_content(url: str) -> Dict[str, Any]:
    """
    Fetches and cleans the main text content of a given URL.
    Use this ONLY during the deep research workflow when provided with a specific URL
    to get its full text content for synthesis.

    Args:
        url: The URL of the web page to fetch.

    Returns:
        A dictionary containing the cleaned content under 'content' (limited length)
        and the source url under 'source', or an error message under 'content'.
    """
    try:
        # Basic URL validation
        if not isinstance(url, str) or not url.startswith("http"):
            raise ValueError("Invalid URL provided.")

        logger.info(f"Fetching content for URL: {url}")
        loader = WebBaseLoader([url])
        loader.requests_per_second = 2 # Be polite
        loader.requests_kwargs = {'timeout': 15} # Set timeout
        docs = loader.load()

        # Optional: Clean HTML - adjust tags based on needs
        # bs_transformer = BeautifulSoupTransformer()
        # docs_transformed = bs_transformer.transform_documents(
        #     docs, tags_to_extract=["p", "li", "div", "h1", "h2", "h3", "span"]
        # )
        # content = docs_transformed[0].page_content if docs_transformed else ""

        # Simpler approach: Join page content, limit length
        content = ""
        if docs:
            content = docs[0].page_content
            # Clean up excessive whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            content = content[:MAX_CONTENT_LENGTH_PER_DOC] # Limit content length

        if not content:
             logger.warning(f"No substantial content found after loading URL: {url}")
             # Return error structure but indicate no content found rather than fetch error
             return {"content": f"No substantial content found at URL.", "source": url}

        logger.info(f"Successfully fetched content (first 100 chars): {content[:100]}")
        return {"content": content, "source": url}

    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}", exc_info=True)
        # Ensure the return structure matches success case for easier handling downstream
        return {"content": f"Error fetching content: {str(e)}", "source": url}


# --- Model Configuration ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-exp-03-25", # Use a powerful model for routing and synthesis
    temperature=0.5,
    api_key=os.environ["GEMINI_API_KEY"],
    convert_system_message_to_human=True, # Often helps Gemini
    # max_tokens=4096 # Let API manage for now
)

# Bind tools that the core agent might call directly
# Note: fetch_web_page_content is also included here, as the 'prepare_fetch_calls'
# node will format calls for it, which are then executed by the ToolNode.
tools = [retrieve_and_process_documents, search_google_places, fetch_web_page_content]
llm_with_tools = llm.bind_tools(tools)

# --- Graph Nodes ---

def call_model(state: GraphState) -> GraphState:
    """Invokes the LLM with the current message history."""
    logger.info("Node: call_model")
    messages = state['messages']
    # Add system prompt only if it's the first message or not present
    if not any(isinstance(m, SystemMessage) for m in messages):
         messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    elif isinstance(messages[0], SystemMessage) and messages[0].content != SYSTEM_PROMPT:
         messages[0] = SystemMessage(content=SYSTEM_PROMPT) # Ensure latest prompt

    response = llm_with_tools.invoke(messages)
    logger.info(f"Model response type: {type(response)}")
    if response.tool_calls:
         logger.info(f"Model requested tool calls: {response.tool_calls}")
    # Append the AI response to the messages list
    return {"messages": [response]}

def plan_research(state: GraphState) -> GraphState:
    """Uses LLM to break down the research topic into sub-queries."""
    logger.info("Node: plan_research")
    user_query = state['messages'][-1].content # Assuming last message is user query triggering research
    core_topic = re.sub(r"^SYSTEM_COMMAND:.*?User research topic:", "", user_query, flags=re.IGNORECASE | re.DOTALL).strip()
    prompt = f"""Act as an expert research strategist. Your task is to break down the following complex research topic into a set of specific, answerable sub-questions that will yield comprehensive search results.

    Research Topic: "{core_topic}"

    Instructions:
    1.  **Identify Key Facets:** Analyze the topic and identify 3-5 (you're free to identify even more if necessary) distinct facets or angles necessary for a thorough understanding. Consider aspects like:
        *   Definition / Background / History
        *   Key Mechanisms / Processes / Technologies Involved
        *   Primary Arguments / Perspectives / Use Cases
        *   Evidence / Data / Case Studies
        *   Challenges / Criticisms / Limitations / Controversies
        *   Current Status / Recent Developments
        *   Future Outlook / Potential Solutions / Implications
    2.  **Formulate Search Queries:** For each identified facet, formulate a concise and effective search engine query. These queries should be designed to find high-quality, informative sources (academic papers, reputable news articles, expert analyses). Avoid overly broad or ambiguous terms.
    3.  **Output Format:** Return ONLY a JSON-formatted list of the search engine query strings. Do not include the facet descriptions or any other text outside the JSON list.

    Example Output Format:
    ["definition of {core_topic}", "mechanisms underlying {core_topic}", "case studies {core_topic} impact", "controversies surrounding {core_topic}", "future predictions {core_topic}"]

    Generate the JSON list of search queries for the research topic: "{core_topic}"
    """
    try:
        planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.5, api_key=os.environ["GEMINI_API_KEY"])
        response = planner_llm.invoke([HumanMessage(content=prompt)])
        plan_str = response.content.strip()
        logger.info(f"Research plan raw response: {plan_str}")

        # Robust JSON parsing
        try:
            # Find the JSON list within the response string
            match = re.search(r'\[.*\]', plan_str, re.DOTALL)
            if match:
                plan_list = json.loads(match.group(0))
                if isinstance(plan_list, list) and all(isinstance(item, str) for item in plan_list):
                    logger.info(f"Generated research plan: {plan_list}")
                    return {"research_plan": plan_list}
                else:
                    raise ValueError("Parsed JSON is not a list of strings.")
            else:
                raise ValueError("No JSON list found in the planner response.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse research plan: {e}. Falling back to using the original query.")
            # Fallback: Use the original query as the only plan item
            return {"research_plan": [user_query]}

    except Exception as e:
        logger.error(f"Error in plan_research LLM call: {e}", exc_info=True)
        # Fallback: Use the original query
        return {"research_plan": [user_query]}


def execute_searches(state: GraphState) -> GraphState:
    """Executes Tavily search for each query in the research plan."""
    logger.info("Node: execute_searches")
    plan = state.get("research_plan", [])
    all_results = []
    if not plan:
        logger.warning("No research plan found. Skipping search execution.")
        return {"search_results": []}

    for query in plan:
        try:
            # Directly invoke the tool's underlying function for simplicity here
            # We are not asking the LLM to decide *to* search here, we *are* searching.
            search_result = retrieve_and_process_documents.invoke({"query": query})
            # Ensure result is a dictionary and has 'documents' key
            if isinstance(search_result, dict) and "documents" in search_result:
                 # Add the query that generated these results for context
                 search_result["query"] = query
                 all_results.append(search_result)
            else:
                 logger.warning(f"Search for query '{query}' did not return expected format. Result: {search_result}")
                 # Add placeholder if needed
                 all_results.append({"query": query, "documents": [{"title": "Error", "source": "", "content": "Failed to retrieve results for this query."}]})

        except Exception as e:
            logger.error(f"Error executing search for query '{query}': {e}", exc_info=True)
            all_results.append({"query": query, "documents": [{"title": "Error", "source": "", "content": f"Exception during search: {e}"}]})

    logger.info(f"Aggregated {len(all_results)} search result sets.")
    return {"search_results": all_results}


def select_urls(state: GraphState) -> GraphState:
    """Selects unique, valid URLs from search results for fetching."""
    logger.info("Node: select_urls")
    search_results = state.get("search_results", [])
    selected_urls = set() # Use a set to automatically handle duplicates

    if not search_results:
        logger.warning("No search results found to select URLs from.")
        return {"selected_urls": []}

    for result_set in search_results:
        documents = result_set.get("documents", [])
        if isinstance(documents, list): # Check if documents is a list
            for doc in documents:
                 # Check if doc is a dictionary and has 'source'
                if isinstance(doc, dict):
                    url = doc.get("source")
                    # Basic validation: must be a string starting with http
                    if isinstance(url, str) and url.startswith("http"):
                        selected_urls.add(url)
                else:
                    logger.warning(f"Skipping invalid document format in results: {doc}")
        elif isinstance(documents, str): # Handle case where 'documents' might be an error string
             logger.warning(f"Encountered error string instead of document list for query '{result_set.get('query')}': {documents}")


    # Limit the number of URLs to fetch
    final_urls = list(selected_urls)[:MAX_SEARCH_RESULTS_TO_FETCH]
    logger.info(f"Selected {len(final_urls)} unique URLs for fetching: {final_urls}")
    return {"selected_urls": final_urls}


def prepare_fetch_calls(state: GraphState) -> GraphState:
    """ Prepares ToolCall objects for the fetch_web_page_content tool. """
    logger.info("Node: prepare_fetch_calls")
    urls_to_fetch = state.get("selected_urls", [])
    messages = state['messages']
    if not urls_to_fetch:
        logger.warning("No URLs selected for fetching.")
        # If no URLs, we can't proceed with fetching, maybe route differently?
        # For now, return state, but synthesis will likely fail.
        # A better approach might be to add a message indicating no URLs found.
        return state # Or add an error message?

    tool_calls = []
    for i, url in enumerate(urls_to_fetch):
        tool_calls.append(
            {
                "id": f"fetch_call_{i}_{uuid.uuid4()}", # Unique ID for each call
                "name": "fetch_web_page_content",
                "args": {"url": url},
            }
        )

    # We need to add an AI message that *looks* like it generated these tool calls
    # so the ToolNode can process them.
    fetch_request_message = AIMessage(
        content="", # No textual content needed, just the calls
        tool_calls=tool_calls,
        # Ensure usage_metadata is None or valid if your model populates it
        usage_metadata=None
    )

    logger.info(f"Prepared {len(tool_calls)} tool calls for fetching.")
    # Append this artificial message to the state
    return {"messages": [fetch_request_message]}


def synthesize_research(state: GraphState) -> GraphState:
    """Synthesizes fetched content into a final research summary."""
    logger.info("Node: synthesize_research")
    messages = state['messages']
    user_query = ""
    # Find the original user query (usually the first HumanMessage)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # Extract fetched content from the ToolMessages added by the ToolNode
    fetched_content_docs = []
    total_length = 0
    for msg in reversed(messages): # Look from the end
        if isinstance(msg, ToolMessage) and msg.tool_call_id.startswith("fetch_call_"):
            try:
                # Content should be the direct string output of the tool
                content_data = json.loads(msg.content) # Tool output is JSON string
                content = content_data.get("content", "")
                source = content_data.get("source", "Unknown Source")

                # Only include successfully fetched content
                if source != "Unknown Source" and not content.startswith("Error"):
                    fetched_content_docs.append({"content": content, "source": source})
                    total_length += len(content)
                    logger.info(f"Included content from: {source} (length: {len(content)})")
                else:
                    logger.warning(f"Skipping content from {source} due to error or missing data: {content}")

            except json.JSONDecodeError:
                 logger.error(f"Failed to decode ToolMessage content: {msg.content}")
            except Exception as e:
                 logger.error(f"Error processing ToolMessage {msg.tool_call_id}: {e}")

    if not fetched_content_docs:
        logger.warning("No valid fetched content found to synthesize.")
        # Return a message indicating failure
        return {"final_summary": "Could not retrieve sufficient information to provide a research summary."}

    logger.info(f"Synthesizing based on {len(fetched_content_docs)} documents, total length approx {total_length} chars.")

    # Simple "Stuff" method for synthesis (concatenate and summarize)
    # WARNING: This might exceed context limits for large amounts of fetched text.
    # Consider MapReduce or Refine for more robust handling if needed.
    context_str = "\n\n---\n\n".join([
        f"Source URL: {doc['source']}\n\nContent:\n{doc['content']}"
        for doc in fetched_content_docs
    ])

    # Limit total context fed to synthesis model
    context_str = context_str[:MAX_TOTAL_SYNTHESIS_LENGTH]

    synthesis_prompt = f"""Act as a meticulous research analyst. Your task is to synthesize the information from the provided text sources into a coherent, well-structured analytical summary addressing the original research request.

    Original Research Request: "{user_query}"

    You have been provided with content fetched from the following sources:
    {chr(10).join([f'- {doc["source"]}' for doc in fetched_content_docs])}

    Combined Content (potentially truncated):
    --- START CONTENT ---
    {context_str}
    --- END CONTENT ---

    Synthesis Instructions:
    1.  **Adhere Strictly to Sources:** Base your entire summary *exclusively* on the information present in the "Combined Content" above. Do NOT introduce external knowledge, opinions, or information.
    2.  **Address the Original Request:** Ensure the summary directly answers or addresses the core aspects of the "Original Research Request".
    3.  **Identify Key Themes & Findings:** Analyze the combined content to identify the main themes, arguments, data points, and conclusions presented across the sources.
    4.  **Synthesize, Don't Just List:** Weave the information together logically. Don't just summarize each source individually. Show how different pieces of information relate to each other.
    5.  **Highlight Agreement/Disagreement (If Applicable):** If sources present conflicting information, different perspectives, or corroborating evidence, explicitly point this out in your summary.
    6.  **Structure the Summary:** Organize your response clearly using Markdown formatting:
        *   **Introduction:** State the topic (as derived from the original request) and the scope covered by the provided sources with a well-written but not extremely verbose introduction.
        *   **Key Findings:** Present the major synthesized themes and findings. Use bullet points or paragraphs for clarity. Support *every* significant claim or data point with evidence from the text. While also emphasizing a coherent flow of information.
        *   **Nuances & Contradictions (Optional):** If you identified disagreements or important nuances across sources, discuss them here.
        *   **Conclusion:** Noete the main conclusions that can be drawn from the provided source material in relation to the original request.
    7.  **Cite Meticulously:** Immediately after presenting any piece of information derived from a source, cite the specific source URL in a Markdown format with the links being numbers like this `[1]`. If multiple sources support a single point, you may list them consecutively with the format provided. Accuracy in citation is critical.
    8.  **Direct Start:** Begin the research as if writing an academic paper.

    Produce the synthesized research based *only* on the provided content.
    """

    try:
        # Use a powerful model for synthesis
        synthesis_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.7, api_key=os.environ["GEMINI_API_KEY"])
        response = synthesis_llm.invoke([HumanMessage(content=synthesis_prompt)])
        summary = response.content.strip()
        logger.info("Successfully generated research summary.")
        # Store fetched docs for final formatting/citation
        return {"final_summary": summary, "fetched_docs": fetched_content_docs}
    except Exception as e:
        logger.error(f"Error during synthesis LLM call: {e}", exc_info=True)
        return {"final_summary": f"Error during synthesis: {e}"}


# In app.py

# --- Helper Node to Store Original Query ---
def store_original_query(state: GraphState) -> GraphState:
    """Stores the initial user query for later use in re-planning."""
    logger.info("Node: store_original_query")
    user_message = next((m for m in reversed(state['messages']) if isinstance(m, HumanMessage)), None)
    if user_message:
        # Clean potential prefixes if coming from /deep_research endpoint
        cleaned_query = re.sub(r"^SYSTEM_COMMAND:.*?User research topic:", "", user_message.content, flags=re.IGNORECASE | re.DOTALL).strip()
        return {"original_user_query": cleaned_query}
    return {}

# --- New Node: Evaluate Search Results ---
def evaluate_search_results(state: GraphState) -> GraphState:
    """
    Evaluates the quality of search results and decides whether to proceed,
    revise the plan, or fail.
    """
    logger.info("Node: evaluate_search_results")
    search_results = state.get("search_results", [])
    current_attempts = state.get("replan_attempts", 0)
    plan_length = len(state.get("research_plan", []))
    decision: Literal["proceed", "revise", "fail_hard"] = "fail_hard" # Default to fail

    if not search_results or plan_length == 0:
        logger.warning("No search results or plan found for evaluation.")
        if current_attempts < MAX_REPLAN_ATTEMPTS:
             decision = "revise" # Try revising if possible
        else:
             decision = "fail_hard"
        return {"evaluation_decision": decision, "replan_attempts": current_attempts + 1}

    successful_queries = 0
    total_docs_found = 0
    min_docs_per_query = 1 # Minimum docs needed for a query to be 'successful'

    for result_set in search_results:
        docs = result_set.get("documents", [])
        # Check if docs is a list and contains actual results (not just error messages)
        if isinstance(docs, list) and len(docs) >= min_docs_per_query:
             # Further check if the first doc isn't an error placeholder
             if not (isinstance(docs[0], dict) and docs[0].get("title") == "Error"):
                  successful_queries += 1
                  total_docs_found += len(docs)

    success_ratio = successful_queries / plan_length if plan_length > 0 else 0
    logger.info(f"Search evaluation: {successful_queries}/{plan_length} queries successful. Total docs: {total_docs_found}. Attempt: {current_attempts+1}")

    # --- Evaluation Criteria (Tune these thresholds) ---
    min_success_ratio = 0.8 # At least 50% of queries should yield some results

    if success_ratio >= min_success_ratio and total_docs_found > 0:
        logger.info("Evaluation result: Proceeding with current results.")
        decision = "proceed"
        # Reset attempts on success? Optional, but prevents future retries if subsequent steps fail.
        # return {"evaluation_decision": decision, "replan_attempts": 0}
        return {"evaluation_decision": decision} # Keep attempts for overall limit
    elif current_attempts < MAX_REPLAN_ATTEMPTS:
        logger.warning(f"Evaluation result: Insufficient results (Ratio: {success_ratio:.2f}). Attempting revision.")
        decision = "revise"
    else:
        logger.error(f"Evaluation result: Insufficient results after {MAX_REPLAN_ATTEMPTS} attempts. Failing research.")
        decision = "fail_hard"

    return {"evaluation_decision": decision, "replan_attempts": current_attempts + 1}


# --- New Node: Revise Plan ---
def revise_plan(state: GraphState) -> GraphState:
    """Generates a *new* research plan based on previous failures."""
    logger.info("Node: revise_plan")
    original_plan = state.get("research_plan", [])
    search_results = state.get("search_results", [])
    original_query = state.get("original_user_query", "Unknown topic") # Use stored query
    current_attempts = state.get("replan_attempts", 0)

    failed_queries = []
    if search_results:
        for i, result_set in enumerate(search_results):
            docs = result_set.get("documents", [])
            query = result_set.get("query", original_plan[i] if i < len(original_plan) else "Unknown Query")
            if not isinstance(docs, list) or len(docs) == 0 or (isinstance(docs[0], dict) and docs[0].get("title") == "Error"):
                failed_queries.append(query)

    logger.info(f"Revising plan. Failed/poor queries: {failed_queries}")

    prompt = f"""You are a research strategist revising a failed plan.
    Original Research Topic: "{original_query}"
    Previous Research Plan: {original_plan}
    Queries yielding poor/no results: {failed_queries}
    Attempt Number: {current_attempts}

    Instructions:
    1. Analyze why the previous queries might have failed (e.g., too specific, too broad, wrong keywords).
    2. Generate a *completely new* set of 3-7 specific, answerable search engine queries for the original topic.
    3. Focus on different angles, keywords, or formulations than the previous plan. Do NOT simply re-use or slightly modify the failed queries.
    4. Output ONLY a JSON list of the new search query strings. Example: ["alternative approach to {original_query}", "broader context {original_query}", "{original_query} expert interviews"]

    Generate the new JSON list of search queries:
    """
    try:
        # Use a capable model for re-planning
        reviser_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.5, api_key=os.environ["GEMINI_API_KEY"])
        response = reviser_llm.invoke([HumanMessage(content=prompt)])
        plan_str = response.content.strip()
        logger.info(f"Revised research plan raw response: {plan_str}")

        # Re-use robust JSON parsing logic from plan_research
        try:
            match = re.search(r'\[.*\]', plan_str, re.DOTALL)
            if match:
                new_plan_list = json.loads(match.group(0))
            else:
                match_md = re.search(r'```json\s*(\[.*\])\s*```', plan_str, re.DOTALL)
                if match_md:
                    new_plan_list = json.loads(match_md.group(1))
                else:
                    raise ValueError("No JSON list found in reviser response.")

            if isinstance(new_plan_list, list) and all(isinstance(item, str) for item in new_plan_list):
                logger.info(f"Generated revised research plan: {new_plan_list}")
                # Crucially, return the *new* plan under the 'research_plan' key
                return {"research_plan": new_plan_list}
            else:
                raise ValueError("Parsed JSON is not a list of strings.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse revised research plan: {e}. Cannot proceed with revision.")
            # If parsing fails, we can't generate a new plan, so force failure.
            # Update state to reflect this hard failure.
            return {"evaluation_decision": "fail_hard"} # Override previous decision

    except Exception as e:
        logger.error(f"Error in revise_plan LLM call: {e}", exc_info=True)
        # Force failure if LLM call errors out
        return {"evaluation_decision": "fail_hard"}

# --- Optional Node: Handle Failed Research ---
def handle_failed_research(state: GraphState) -> GraphState:
    """Creates a message indicating research failure after retries."""
    logger.error("Node: handle_failed_research - Research failed after maximum retries.")
    original_query = state.get("original_user_query", "the requested topic")
    failure_message = AIMessage(
        content=f"I apologize, but I encountered difficulties gathering sufficient information for '{original_query}' even after attempting different search strategies. Please try rephrasing your request or asking about a different topic."
    )
    # Add this message to the history. format_final_response will pick it up.
    # We could also set final_summary directly, but adding to messages is cleaner.
    return {"messages": [failure_message]}



# --- Routing Logic ---

class RouteQuerySchema(BaseModel):
    """Schema for the routing decision."""
    action: Literal["research", "search_web", "search_places", "respond_directly"] = Field(
        ...,
        description="The next action to take based on the user query. Choose 'research' for in-depth investigation requests. Choose 'search_web' for factual questions needing current info or web lookup. Choose 'search_places' for location-specific queries. Choose 'respond_directly' if the query can be answered from general knowledge or conversation history."
    )

def route_query(state: GraphState) -> Literal["research", "tools", "respond"]:
    """Routes the query to research, tools (web/places search), or direct response."""
    logger.info("Node: route_query")
    messages = state['messages']
    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_user_message:
        logger.warning("No user message found for routing.")
        return "respond" # Default to direct response if no user query

    try:
        # Use a smaller/faster model for routing
        router_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, api_key=os.environ["GEMINI_API_KEY"])
        structured_llm = router_llm.with_structured_output(RouteQuerySchema)

        routing_prompt = f"""Given the user's latest message and the assistant's capabilities (search engine, tutor, vacation advisor, researcher), determine the best course of action.

        User Query: "{last_user_message.content}"

        Consider keywords like 'research', 'investigate', 'summarize findings on' for the 'research' action.
        Consider questions about facts, current events, or general knowledge needing external lookup for 'search_web'.
        Consider queries mentioning locations, businesses, addresses, 'near me' for 'search_places'.
        If the query is conversational, a follow-up, or can be answered directly, choose 'respond_directly'.

        What is the appropriate next action?
        """
        # Pass only the last user message for efficiency
        route_decision = structured_llm.invoke([HumanMessage(content=routing_prompt)])

        logger.info(f"Routing decision: {route_decision.action}")

        if route_decision.action == "research":
            return "research"
        elif route_decision.action in ["search_web", "search_places"]:
            if route_decision.action == "research":
                 return "research" # Start the research plan
            else:
                 # If the router suggests search/places/direct, but the main model didn't
                 # make a tool call, we assume it's a direct response.
                 return "respond" # Go to end

        elif route_decision.action == "respond_directly":
            return "respond" # Go to end

    except Exception as e:
        logger.error(f"Error in route_query: {e}", exc_info=True)
        return "respond" # Default to direct response on error

# --- Build Graph ---
# In app.py (after all node functions like call_model, plan_research, evaluate_search_results, etc., are defined)

# --- Helper Functions for Graph Logic ---

def route_query_updater(state: GraphState) -> GraphState:
    """Wrapper for route_query to update state with the decision."""
    # Assuming route_query function is defined above and returns a string like "research" or "respond"
    decision = route_query(state) # Calls the LLM-based routing logic
    logger.info(f"route_query_updater: Storing decision '{decision}' into state.")
    return {"route_decision": decision}

def should_continue_or_route(state: GraphState) -> Literal["tools", "route_query_node"]:
    """Decide whether to call tools based on last message, or route query if no tools called."""
    if not state['messages']:
        # Should not happen with entry point, but safeguard
        return "route_query_node"
    last_message = state['messages'][-1]

    # If the agent produced tool calls, execute them
    if getattr(last_message, "tool_calls", None):
         logger.info("Edge: Agent -> Tools (Tool call detected)")
         return "tools"
    else:
        # Otherwise, route the query to decide if research is needed or respond directly
        logger.info("Edge: Agent -> route_query_node (No tool call detected)")
        return "route_query_node"

def route_after_tools(state: GraphState) -> Literal["synthesize", "agent"]:
     """ Routes to synthesis if fetch tool calls were just executed, otherwise back to agent."""
     logger.info("Node: route_after_tools check")
     if not state['messages']:
          logger.warning("route_after_tools: No messages found in state, routing to agent.")
          return "agent"

     last_message = state['messages'][-1]
     # Check if the *last* message is a ToolMessage resulting from a fetch call
     tool_call_id = getattr(last_message, 'tool_call_id', None)
     if isinstance(last_message, ToolMessage) and tool_call_id and tool_call_id.startswith("fetch_call_"):
          logger.info("Edge: Tools -> synthesize_research (Detected fetch result)")
          return "synthesize"
     else:
          # Assume other tool calls (search, places) or non-tool messages should go back to agent
          logger.info(f"Edge: Tools -> agent (Last message type: {type(last_message).__name__})")
          return "agent"

# --- Build Graph ---

workflow = StateGraph(GraphState)

# 1. Add Nodes
# Core Agent & Tools
workflow.add_node("agent", call_model)
tool_node = ToolNode(tools) # Executes tools listed in 'tools' variable
workflow.add_node("tools", tool_node)
workflow.add_node("route_query_node", route_query_updater) # Node that calls the LLM router and updates state

# Research Planning & Execution Flow
workflow.add_node("store_original_query", store_original_query)
workflow.add_node("plan_research", plan_research)
workflow.add_node("execute_searches", execute_searches)

# Adaptive Re-planning Flow
workflow.add_node("evaluate_search_results", evaluate_search_results)
workflow.add_node("revise_plan", revise_plan)
workflow.add_node("handle_failed_research", handle_failed_research)

# Content Fetching & Synthesis Flow
workflow.add_node("select_urls", select_urls)
workflow.add_node("prepare_fetch_calls", prepare_fetch_calls) # Creates fetch tool calls
workflow.add_node("synthesize_research", synthesize_research) # Runs MapReduce chain

# Note: route_after_tools is used directly in conditional edge, no separate node needed unless adding specific logic/logging

# 2. Define Edges

# Entry Point
workflow.set_entry_point("agent")

# Agent Decision Point
workflow.add_conditional_edges(
    "agent",
    should_continue_or_route, # Checks if agent requested tools
    {
        "tools": "tools",                 # If yes, execute tools
        "route_query_node": "route_query_node" # If no, route the query
    }
)

# Query Routing Decision Point
workflow.add_conditional_edges(
    "route_query_node",
    lambda x: x.get('route_decision', 'respond'), # Read decision from state
    {
        "research": "store_original_query", # Start research workflow
        "respond": END,                      # End if direct response needed
        # Add handling for search_web/search_places if route_query could return those
        # Currently assumes agent handles search tool calls directly
    }
)

# Research Workflow Path
workflow.add_edge("store_original_query", "plan_research")
workflow.add_edge("plan_research", "execute_searches")
workflow.add_edge("execute_searches", "evaluate_search_results") # Always evaluate after searching

# Evaluation Decision Point (Adaptive Re-planning)
workflow.add_conditional_edges(
    "evaluate_search_results",
    lambda x: x.get('evaluation_decision', 'fail_hard'), # Read evaluation decision
    {
        "proceed": "select_urls",        # Good results -> continue to URL selection
        "revise": "revise_plan",         # Bad results, retries left -> revise plan
        "fail_hard": "handle_failed_research" # Bad results, no retries -> handle failure
    }
)

# Re-planning Loop
workflow.add_edge("revise_plan", "execute_searches") # After revising, search again

# Failure Path
workflow.add_edge("handle_failed_research", END) # End after handling failure

# Successful Research - Content Fetching Path
workflow.add_edge("select_urls", "prepare_fetch_calls")
workflow.add_edge("prepare_fetch_calls", "tools") # Send fetch calls to ToolNode

# Post-Tool Execution Routing
workflow.add_conditional_edges(
    "tools",
    route_after_tools, # Checks if fetch calls were just run
    {
        "synthesize": "synthesize_research", # Fetch results go to synthesis
        "agent": "agent"                     # Other tool results go back to agent
    }
)

# Final Research Step
workflow.add_edge("synthesize_research", END)

# 3. Initialize Memory Saver (Optional but recommended for conversation history)
checkpointer = MemorySaver()

# 4. Compile the Graph
# Add checkpointer for memory persistence across requests
app = workflow.compile(checkpointer=checkpointer)

# Optional: Assign to a standard name if used elsewhere (e.g., in API)
app_instance = app

# --- Response Formatting ---

def format_final_response(app_result: Dict[str, Any]) -> str:
    """
    Formats the final response, prioritizing research summary if available,
    otherwise formatting the last AI message and adding sources.
    """
    try:
        state = app_result # The final state dict after graph execution

        # --- 1. Check for Research Summary ---
        final_summary = state.get("final_summary")
        fetched_docs = state.get("fetched_docs") # List of {'content': '...', 'source': '...'}

        if final_summary and fetched_docs:
            logger.info("Formatting research summary response.")
            sources_section = "\n\n**Sources Consulted:**\n"
            for i, doc in enumerate(fetched_docs, 1):
                url = doc.get("source", "Source URL not found")
                # Basic validation
                if not isinstance(url, str) or not url.startswith("http"):
                    url = "Invalid or missing URL"
                sources_section += f"[{i}] {url}\n"

            # Ensure summary doesn't already contain the sources section header
            summary_content = final_summary.strip()
            if "**Sources Consulted:**" not in summary_content:
                 final_output = f"**Research Findings:**\n\n{summary_content}{sources_section}"
            else:
                 # Assume summary already includes sources if header is present
                 final_output = summary_content
            return final_output.strip()

        # --- 2. Format Standard AI Response (if no research summary) ---
        logger.info("Formatting standard AI response.")
        messages = state.get("messages", [])
        final_answer_content = "Could not determine the final answer."
        tool_outputs = [] # To collect outputs from search/places tools

        # Find the last AI message intended for the user
        # This might be tricky if the last message was the one initiating tool calls
        # Look for the last AIMessage *without* tool calls, or the one *before* the last ToolMessages
        last_ai_message_for_user = None
        for msg in reversed(messages):
             if isinstance(msg, AIMessage) and not msg.tool_calls:
                  last_ai_message_for_user = msg
                  break
        # If not found, maybe the last AI message *with* tool calls was meant to contain text?
        if not last_ai_message_for_user:
             for msg in reversed(messages):
                  if isinstance(msg, AIMessage):
                       last_ai_message_for_user = msg
                       break

        if last_ai_message_for_user:
             final_answer_content = last_ai_message_for_user.content.strip()

        # Extract sources/info from ToolMessages (non-fetch ones)
        retrieved_docs_sources = []
        places_info_used = False
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = ""
                # Find the corresponding tool call ID in previous AI messages
                for prev_msg in reversed(messages):
                     if isinstance(prev_msg, AIMessage) and prev_msg.tool_calls:
                          for tc in prev_msg.tool_calls:
                               if tc.get('id') == msg.tool_call_id:
                                    tool_name = tc.get('name')
                                    break
                          if tool_name: break

                if tool_name == "retrieve_and_process_documents":
                    try:
                        data = json.loads(msg.content)
                        docs = data.get("documents", [])
                        if isinstance(docs, list):
                             for doc in docs:
                                  if isinstance(doc, dict):
                                       title = doc.get("title", "Untitled")
                                       url = doc.get("source", "No URL")
                                       if url.startswith("http"): # Add only valid URLs
                                            retrieved_docs_sources.append({"title": title, "url": url})
                        elif isinstance(docs, str): # Handle error string
                             logger.warning(f"Tool message for retrieve_and_process_documents contained error: {docs}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode tool message content for retrieve_and_process_documents: {msg.content}")
                elif tool_name == "search_google_places":
                     try:
                          data = json.loads(msg.content)
                          results = data.get("places_results")
                          # Check if results are not an error message
                          if results and not (isinstance(results, str) and results.startswith("Error")):
                               places_info_used = True
                     except json.JSONDecodeError:
                          # If content isn't JSON, it might be a direct string result/error
                          if not msg.content.startswith("Error"):
                               places_info_used = True
                          logger.warning(f"Tool message content for search_google_places was not JSON: {msg.content}")


        # Build sources section for standard response
        sources_section = ""
        unique_sources = {src['url']: src['title'] for src in retrieved_docs_sources} # Deduplicate URLs
        if unique_sources:
            sources_section = "\n\n**Sources:**\n"
            for i, (url, title) in enumerate(unique_sources.items(), 1):
                sources_section += f"[{i}] [{title}]({url})\n"
        elif places_info_used:
            sources_section = "\n\n*(Used external location search to provide this information)*"
        else:
            sources_section = "\n" # Ensure newline separation

        # Combine content and sources
        final_output = f"{final_answer_content}{sources_section}"
        return final_output.strip()

    except Exception as e:
        logger.error(f"Error formatting final response: {e}", exc_info=True)
        # Fallback: Try to return the last message content directly
        try:
            state = app_result
            messages = state.get("messages", [])
            if messages:
                return messages[-1].content
            return "Error during response formatting."
        except Exception:
            return "Critical error during response formatting and fallback."

