from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage # Keep only necessary message types
import logging
import json
import uuid
from typing import List, Optional, Dict, Any
import base64 # <-- Added
import mimetypes # <-- Added
import uuid
from typing import List, Optional, Dict, Any
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Body, # Keep for potential future JSON endpoints
    Form, # <-- Added
    File, # <-- Added
    UploadFile, # <-- Added
)

MAX_FILE_SIZE_MB = 10
MAX_TOTAL_UPLOAD_SIZE_MB = 20
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
    "audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4",
    "video/mp4", "video/quicktime", "video/webm", "video/mpeg",
}
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TOTAL_UPLOAD_SIZE = MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024


# Import the compiled graph instance and the formatting function
# Ensure app.py is in the same directory or accessible via PYTHONPATH
try:
    from app import app_instance, format_final_response
except ImportError:
    print("Error: Could not import 'app_instance' or 'format_final_response' from app.py.")
    print("Ensure app.py exists and is runnable.")
    exit() # Exit if core components can't be imported

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Initialize FastAPI app
api_app = FastAPI()

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)
api_app.state.limiter = limiter

# Add the SlowAPI middleware
api_app.add_middleware(SlowAPIMiddleware)

# CORS Middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sensus_api")

# --- Pydantic Models ---
# Keep BaseRequest for potential future use if needed, but endpoints now use Form/File
class BaseRequest(BaseModel):
    thread_id: Optional[str] = Field(None, description="Optional conversation thread ID.", example="thread_12345")

# QueryRequest and ResearchRequest are no longer used directly by endpoints accepting form data
# class QueryRequest(BaseRequest): ...
# class ResearchRequest(BaseRequest): ...

class ApiResponse(BaseModel):
    response: str
    thread_id: str

# --- Helper Functions ---

async def process_uploaded_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Reads, validates, encodes, and structures uploaded files for HumanMessage."""
    media_content_blocks = []
    total_size = 0
    if not files:
        return []
    for file in files:
        if not file or file.filename == "": continue
        file_bytes = await file.read()
        file_size = len(file_bytes)
        total_size += file_size
        if file_size == 0:
            logger.warning(f"Skipping empty file: {file.filename}")
            continue
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File '{file.filename}' ({file_size / 1024 / 1024:.2f} MB) exceeds max size {MAX_FILE_SIZE_MB} MB.")
        if total_size > MAX_TOTAL_UPLOAD_SIZE:
             raise HTTPException(status_code=413, detail=f"Total upload size exceeds limit {MAX_TOTAL_UPLOAD_SIZE_MB} MB.")
        mime_type = file.content_type
        if not mime_type: mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type or mime_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=415, detail=f"File '{file.filename}' has unsupported MIME type: '{mime_type}'. Allowed: {', '.join(ALLOWED_MIME_TYPES)}")
        base64_encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{base64_encoded_data}"
        content_type_key = None
        if mime_type.startswith("image/"): content_type_key = "image_url"
        elif mime_type.startswith("audio/"): content_type_key = "audio_url"
        elif mime_type.startswith("video/"): content_type_key = "video_url"
        if content_type_key:
            media_content_blocks.append({"type": content_type_key, content_type_key: {"url": data_uri}})
            logger.info(f"Processed file: {file.filename} ({mime_type}), size: {file_size} bytes")
        else: logger.warning(f"Could not determine content block type for MIME type: {mime_type}")
    return media_content_blocks

async def invoke_graph(input_message: HumanMessage, thread_id: Optional[str]) -> dict:
    """Handles invoking the graph and basic state validation."""
    # (This function remains largely the same as your original, just ensure logging is helpful)
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Processing request for thread_id: {thread_id}")

    # Log message content structure (abbreviated)
    log_content = []
    if isinstance(input_message.content, str):
        log_content.append({"type": "text", "text_preview": input_message.content[:100] + "..."})
    elif isinstance(input_message.content, list):
        for item in input_message.content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    log_content.append({"type": "text", "text_preview": item.get("text", "")[:100] + "..."})
                elif item_type in ["image_url", "audio_url", "video_url"]:
                    url_value = item.get(item_type, {}).get("url", "")
                    log_content.append({"type": item_type, "data_uri_preview": url_value[:80] + "..."})
                else: log_content.append({"type": item_type})
            else: log_content.append({"type": "unknown_item_format"})
    logger.info(f"Input message structure: {log_content}")

    input_dict = {"messages": [input_message]}
    logger.info("Invoking graph...")
    final_state = await app_instance.ainvoke(input_dict, config=config)
    logger.info("Graph invocation complete.")

    if not final_state or not isinstance(final_state, dict):
        logger.error(f"Graph execution returned invalid state type for thread_id: {thread_id}. Type: {type(final_state)}")
        raise HTTPException(status_code=500, detail="Agent failed to generate a valid internal state.")
    if "messages" not in final_state or not final_state["messages"]:
         logger.error(f"Graph execution state missing 'messages' or list is empty for thread_id: {thread_id}")
         raise HTTPException(status_code=500, detail="Agent failed to generate response messages.")

    # Optional logging of results remains the same
    if "final_summary" in final_state and final_state["final_summary"]: logger.info(f"State contains final_summary for thread_id: {thread_id}")
    # ... other result logging ...

    return final_state, thread_id


# --- API Endpoints ---

@api_app.post("/query", response_model=ApiResponse, tags=["Querying"])
@limiter.limit("10/minute")
async def query_endpoint(
    request: Request,
    # Use Form for text/thread_id, File for uploads instead of Body(QueryRequest)
    text_input: str = Form(..., description="The user's question or statement.", alias="query"), # Use alias if frontend sends 'query'
    thread_id: Optional[str] = Form(None, description="Optional conversation thread ID."),
    media_files: List[UploadFile] = File([], description="Optional list of media files.") # Default to empty list
):
    """
    Handles general user queries, potentially including media files.
    Accepts multipart/form-data.
    """
    try:
        # 1. Process uploaded files
        media_blocks = await process_uploaded_files(media_files)

        # 2. Construct multimodal message content
        content_list = [{"type": "text", "text": text_input}]
        content_list.extend(media_blocks)
        input_message = HumanMessage(content=content_list)

        # 3. Invoke graph
        final_state, returned_thread_id = await invoke_graph(input_message, thread_id)

        # 4. Format response
        logger.info(f"Formatting final response for thread_id: {returned_thread_id} (standard query)")
        formatted_response = format_final_response(final_state)
        logger.info(f"Final Formatted Response Length: {len(formatted_response)}")

        return ApiResponse(response=formatted_response, thread_id=returned_thread_id)

    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for /query from {request.client.host}: {e.detail}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded") from e
    except HTTPException as e:
        raise e # Reraise validation (413, 415), rate limit (429), or internal (500) errors
    except Exception as e:
        error_message = f"Unhandled error during /query processing: {e}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@api_app.post("/deep_research", response_model=ApiResponse, tags=["Querying"])
@limiter.limit("3/minute")
async def deep_research_endpoint(
    request: Request,
    # Use Form for text/thread_id, File for uploads instead of Body(ResearchRequest)
    topic: str = Form(..., description="The topic for in-depth research."),
    thread_id: Optional[str] = Form(None, description="Optional conversation thread ID."),
    media_files: List[UploadFile] = File([], description="Optional media files providing context.") # Default to empty list
):
    """
    Triggers deep research, potentially using media files for context.
    Accepts multipart/form-data.
    """
    try:
        # 1. Prepare text content
        research_instruction = "SYSTEM_COMMAND: Initiate deep research workflow."
        text_content = f"{research_instruction}\n\nUser research topic: {topic}"

        # 2. Process uploaded files
        media_blocks = await process_uploaded_files(media_files)

        # 3. Construct multimodal message content
        content_list = [{"type": "text", "text": text_content}]
        content_list.extend(media_blocks)
        input_message = HumanMessage(content=content_list)

        # 4. Invoke graph
        final_state, returned_thread_id = await invoke_graph(input_message, thread_id)

        # 5. Format response
        logger.info(f"Formatting final response for thread_id: {returned_thread_id} (deep research)")
        formatted_response = format_final_response(final_state)
        logger.info(f"Final Formatted Response Length: {len(formatted_response)}")

        # Optional check for failure messages remains the same
        if "Could not retrieve sufficient information" in formatted_response or "Error during synthesis" in formatted_response:
             logger.warning(f"Deep research for thread_id {returned_thread_id} may have failed or yielded no results.")

        return ApiResponse(response=formatted_response, thread_id=returned_thread_id)

    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for /deep_research from {request.client.host}: {e.detail}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded") from e
    except HTTPException as e:
        raise e
    except Exception as e:
        error_message = f"Unhandled error during /deep_research processing: {e}"
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# --- Middleware and Exception Handlers ---

@api_app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url} from {request.client.host}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error during request processing middleware: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "An internal server error occurred in middleware."})

@api_app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@api_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@api_app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"message": getattr(exc, 'detail', "Rate limit exceeded.")})

# Add specific handler for HTTPException to log different levels
@api_app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code in [413, 415, 429, 422]:
         logger.warning(f"HTTP Exception {exc.status_code} for {request.url}: {exc.detail}")
    elif exc.status_code >= 500:
         logger.error(f"HTTP Server Error {exc.status_code} for {request.url}: {exc.detail}", exc_info=False)
    else:
         logger.info(f"HTTP Client Error {exc.status_code} for {request.url}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@api_app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catch unexpected errors
    if not isinstance(exc, HTTPException): # Avoid double handling if an HTTPException wasn't caught above
        logger.error(f"Unhandled internal server error during request to {request.url}: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "An internal server error occurred."})
    # If it was an HTTPException somehow missed, return its response
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
