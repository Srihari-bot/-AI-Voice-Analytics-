from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
import tempfile
import logging
import time
from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from typing import Optional
from Intent_Prompt import get_intent_system_message, get_intent_user_message
from rag_pipeline import knowledge_retriever
from Resolution_Prompt import get_resolution_user_message_with_rag, format_tally_resolution, get_resolution_system_message
from AI_keyword import get_keyword_extraction_system_message, get_keyword_extraction_user_message, validate_extraction_response, extract_keywords_simple_fallback
import chromadb
from chromadb.config import Settings
import numpy as np
import base64
from uuid import uuid4
import asyncio
from functools import wraps
from pydub import AudioSegment
from io import BytesIO
import re
from groq import Groq

import os
os.environ["CHROMA_TELEMETRY"] = "false"           # ✅ CORRECT
os.environ["ANONYMIZED_TELEMETRY"] = "false"       # ✅ CORRECT  
os.environ["POSTHOG_DISABLED"] = "true" 

# Now import your modules
from rag_pipeline import knowledge_retriever
# ... rest of your imports


# Load environment variables
load_dotenv()

# Function to reload environment variables
def reload_env_variables():
    """Reload environment variables from .env file."""
    load_dotenv(override=True)
    logger.info("Environment variables reloaded from .env file")

# Function to get environment variable with reload capability
def get_env_var(key: str, default: str = None, reload: bool = False) -> str:
    """Get environment variable with optional reload."""
    if reload:
        reload_env_variables()
    return os.getenv(key, default)

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "audio_kb")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
MAX_FILE_MB = 30

# Retry configuration
MAX_RETRIES = 5  # Maximum 5 retry attempts
BASE_DELAY = 1  # Base delay in seconds
MAX_DELAY = 60  # Maximum delay in seconds
TIMEOUT_SECONDS = 300  # 5 minutes default timeout

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="GST & License Analyzer API",
    description="API for audio transcription (using Kibo AI) and GST/License analysis using IBM WatsonX",
    version="1.0.0"
)

# CORS middleware - Updated to match frontend port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # React Vite development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload configuration
ALLOWED_EXT = [".mp3", ".wav"]

# Global variables for token and model management
access_token_cache = {
    "token": None,
    "expiry": 0,
    "api_key_hash": None  # Track API key to detect changes
}

def clear_token_cache():
    """Clear cached access token and force refresh."""
    global access_token_cache
    access_token_cache["token"] = None
    access_token_cache["expiry"] = 0
    access_token_cache["api_key_hash"] = None
    logger.info("Access token cache cleared")


def get_api_key_hash(api_key: str) -> str:
    """Generate a simple hash of the API key to detect changes."""
    import hashlib
    return hashlib.md5(api_key.encode()).hexdigest()[:8]

# Initialize ChromaDB client with same settings as RAG pipeline
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIRECTORY,
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

# STT Model Selection State
current_stt_model = "whisper"  # Default to Whisper, can be "kibo", "whisper", or "assisto"

# Startup event to ensure collection exists
@app.on_event("startup")
def startup_event():
    """Initialize ChromaDB collection on startup."""
    ensure_collection()

# Pydantic models
class TranscriptionResponse(BaseModel):
    transcription: str
    success: bool
    message: str

class IntentResponse(BaseModel):
    intent: str
    success: bool
    message: str

class ResolutionResponse(BaseModel):
    resolution: str
    success: bool
    message: str

class KeywordExtractionResponse(BaseModel):
    keywords: list
    overall_confidence: int
    success: bool
    message: str

class FullAnalysisResponse(BaseModel):
    transcription: str
    intent: str
    resolution: str
    keywords: dict = None
    success: bool
    message: str

class SearchIn(BaseModel):
    query: str
    top_k: int = 5
    tag: str = None

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Final attempt {attempt + 1} failed: {str(e)}. No more retries will be attempted.")
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} of {max_retries + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            
            # Re-raise the last exception if all retries failed
            logger.error(f"All {max_retries + 1} attempts failed. Giving up.")
            raise last_exception
        return wrapper
    return decorator

# Enhanced error handling for network requests
class APIError(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, message, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

# Global exception handler for custom API errors
@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    """Handle custom API errors and convert them to proper HTTP responses."""
    logger.error(f"API Error: {exc}")
    
    # Determine appropriate HTTP status code
    status_code = exc.status_code or 500
    
    # Handle specific error types
    if status_code == 504:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Gateway Timeout",
                "message": str(exc),
                "retry_after": exc.retry_after,
                "suggestion": "The request took too long. Please try again with a smaller file or wait before retrying."
            }
        )
    elif status_code == 503:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": str(exc),
                "retry_after": exc.retry_after,
                "suggestion": "The service is temporarily unavailable. Please try again later."
            }
        )
    elif status_code == 429:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "message": str(exc),
                "retry_after": exc.retry_after,
                "suggestion": "Rate limit exceeded. Please wait before making another request."
            }
        )
    else:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "API Error",
                "message": str(exc),
                "suggestion": "Please check your request and try again."
            }
        )

# Global exception handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions and provide helpful error messages."""
    logger.error(f"Unhandled exception: {exc}")
    
    # Check if it's a known error type
    if "timeout" in str(exc).lower():
        return JSONResponse(
            status_code=504,
            content={
                "error": "Request Timeout",
                "message": "The request took too long to complete.",
                "suggestion": "Please try again with a smaller file or wait before retrying."
            }
        )
    elif "connection" in str(exc).lower():
        return JSONResponse(
            status_code=503,
            content={
                "error": "Connection Error",
                "message": "Unable to connect to the required service.",
                "suggestion": "Please check your internet connection and try again."
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred.",
                "suggestion": "Please try again later or contact support if the problem persists."
            }
        )

def handle_request_exception(e, api_name="API"):
    """Handle different types of request exceptions and provide appropriate error messages."""
    if isinstance(e, requests.exceptions.Timeout):
        logger.error(f"{api_name} timeout error: {str(e)}")
        raise APIError(f"{api_name} request timed out. The service may be busy or the request is too large.", status_code=504)
    
    elif isinstance(e, requests.exceptions.ConnectionError):
        logger.error(f"{api_name} connection error: {str(e)}")
        raise APIError(f"{api_name} connection failed. Please check your internet connection and try again.", status_code=503)
    
    elif isinstance(e, requests.exceptions.SSLError):
        logger.error(f"{api_name} SSL error: {str(e)}")
        raise APIError(f"{api_name} SSL certificate error. Please check your system's SSL configuration.", status_code=502)
    
    elif isinstance(e, requests.exceptions.RequestException):
        logger.error(f"{api_name} request error: {str(e)}")
        raise APIError(f"{api_name} request failed: {str(e)}", status_code=500)
    
    else:
        logger.error(f"Unexpected {api_name} error: {str(e)}")
        raise APIError(f"Unexpected {api_name} error: {str(e)}", status_code=500)

@retry_with_backoff(max_retries=5, base_delay=2, max_delay=30)
def get_access_token(api_key: str, force_refresh: bool = False) -> Optional[str]:
    """
    Get IBM Cloud access token with enhanced error handling and validation.
    """
    try:
        # Validate API key format
        if not api_key or not isinstance(api_key, str):
            raise HTTPException(status_code=500, detail="Invalid or missing API key")
        
        # Remove any whitespace/newlines from API key
        api_key = api_key.strip()
        
        # Log API key format for debugging (first and last 4 chars only)
        if len(api_key) >= 8:
            logger.info(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
        
        current_time = time.time()
        current_api_key_hash = get_api_key_hash(api_key)
        
        # Check if API key has changed (clear cache if it has)
        if (access_token_cache["api_key_hash"] and 
            access_token_cache["api_key_hash"] != current_api_key_hash):
            logger.info("API key changed, clearing token cache")
            clear_token_cache()
        
        # Check if we have a valid cached token
        if (not force_refresh and 
            access_token_cache["token"] and 
            current_time < access_token_cache["expiry"]):
            logger.info("Using cached access token")
            return access_token_cache["token"]
        
        logger.info("Requesting new access token...")
        
        # IBM Cloud IAM token endpoint
        url = "https://iam.cloud.ibm.com/identity/token"
        
        # Headers - exactly as IBM Cloud expects
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        # Data payload - ensure proper encoding
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key
        }
        
        # Make the request with proper timeout and error handling
        try:
            response = requests.post(
                url, 
                headers=headers, 
                data=data, 
                timeout=30,
                verify=True  # Ensure SSL verification
            )
            
            # Log response details for debugging
            logger.info(f"Token request status: {response.status_code}")
            
            # Handle different error scenarios
            if response.status_code == 400:
                error_detail = response.text
                logger.error(f"400 Bad Request - Response: {error_detail}")
                
                # Common causes of 400 errors
                if "BXNIM0415E" in error_detail or "could not be found" in error_detail.lower():
                    raise HTTPException(
                        status_code=500, 
                        detail="IBM Cloud API key not found. Please check your WATSONX_API_KEY environment variable and ensure it's valid."
                    )
                elif "invalid_grant" in error_detail.lower():
                    raise HTTPException(
                        status_code=500, 
                        detail="Invalid API key. Please check your WATSONX_API_KEY environment variable."
                    )
                elif "invalid_request" in error_detail.lower():
                    raise HTTPException(
                        status_code=500, 
                        detail="Invalid token request format. Please check API key configuration."
                    )
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"IBM Cloud authentication failed: {error_detail}"
                    )
            
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=500, 
                    detail="Unauthorized - Invalid API key or insufficient permissions"
                )
            
            elif response.status_code == 403:
                raise HTTPException(
                    status_code=500, 
                    detail="Forbidden - API key may not have required permissions"
                )
            
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"IBM Cloud IAM error: {response.text}"
                )
            
            # Parse successful response
            token_data = response.json()
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            token_type = token_data.get("token_type", "Bearer")
            
            if not access_token:
                logger.error("No access token in successful response")
                logger.error(f"Response data: {token_data}")
                raise HTTPException(
                    status_code=500, 
                    detail="No access token received from IBM Cloud"
                )
            
            # Store token and calculate expiry time (with 5-minute buffer)
            access_token_cache["token"] = access_token
            access_token_cache["expiry"] = current_time + expires_in - 300
            access_token_cache["api_key_hash"] = current_api_key_hash
            
            logger.info(f"New access token obtained successfully")
            logger.info(f"Token type: {token_type}, expires in: {expires_in} seconds")
            
            return access_token
        
        except requests.exceptions.Timeout:
            raise APIError("Timeout connecting to IBM Cloud IAM service", status_code=504)
        except requests.exceptions.ConnectionError:
            raise APIError("Connection error to IBM Cloud IAM service", status_code=503)
        except requests.exceptions.SSLError:
            raise APIError("SSL certificate error connecting to IBM Cloud", status_code=502)
        except requests.exceptions.RequestException as e:
            raise APIError(f"Network error connecting to IBM Cloud: {str(e)}", status_code=500)
    
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except APIError:
        # Re-raise APIErrors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting access token: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during authentication: {str(e)}"
        )

@retry_with_backoff(max_retries=5, base_delay=2, max_delay=30)
def make_watsonx_request(payload: dict, api_key: str, max_retries: int = 2) -> Optional[dict]:
    """
    Make a request to WatsonX with automatic token refresh on 401 errors and enhanced error handling.
    """
    for attempt in range(max_retries + 1):
        try:
            force_refresh = attempt > 0
            # Reload environment variables on first attempt to get latest API key
            if attempt == 0:
                reload_env_variables()
                # Get fresh API key in case it was updated
                fresh_api_key = get_env_var("WATSONX_API_KEY")
                if fresh_api_key and fresh_api_key != api_key:
                    logger.info("API key updated, using fresh key")
                    api_key = fresh_api_key
            access_token = get_access_token(api_key, force_refresh=force_refresh)
            
            if not access_token:
                raise APIError("Failed to obtain access token", status_code=500)
            
            # WatsonX API configuration
            watsonx_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Dynamic timeout based on payload size
            timeout = min(TIMEOUT_SECONDS, max(60, len(str(payload)) // 100))  # 1 second per 100 chars, min 60s, max TIMEOUT_SECONDS
            
            logger.info(f"Making WatsonX request with {timeout}s timeout")
            
            response = requests.post(
                watsonx_url,
                headers=headers,
                json=payload,
                params={"version": "2023-05-29"},
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401 and attempt < max_retries:
                logger.warning(f"Token expired on attempt {attempt + 1}, refreshing...")
                access_token_cache["token"] = None
                access_token_cache["expiry"] = 0
                continue
            elif response.status_code == 429:  # Rate limit
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            elif response.status_code == 503:  # Service unavailable
                logger.warning(f"Service unavailable on attempt {attempt + 1}, retrying...")
                time.sleep(5 * (attempt + 1))  # Progressive delay
                continue
            else:
                logger.error(f"API Response Status: {response.status_code}")
                logger.error(f"API Response: {response.text}")
                raise APIError(
                    f"WatsonX API Error: {response.text}", 
                    status_code=response.status_code
                )
                
        except APIError:
            # Re-raise APIErrors
            raise
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(5 * (attempt + 1))
                continue
            else:
                raise APIError("WatsonX API request timed out after all retries", status_code=504)
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                time.sleep(5 * (attempt + 1))
                continue
            else:
                raise APIError("Failed to connect to WatsonX API after all retries", status_code=503)
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries:
                raise APIError(f"Failed to connect to WatsonX API after {max_retries + 1} attempts: {str(e)}", status_code=500)
            else:
                time.sleep(5 * (attempt + 1))
    
    return None

def get_kibo_api_key(reload: bool = False) -> str:
    """
    Get Kibo API key from environment variables.
    """
    api_key = get_env_var("API_KEY", reload=reload)
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="API_KEY not found in environment variables. Please set API_KEY in your .env file."
        )
    return api_key

def get_whisper_api_key(reload: bool = False) -> str:
    """
    Get Whisper API key from environment variables.
    """
    api_key = get_env_var("WHISPER_API_KEY", reload=reload)
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="WHISPER_API_KEY not found in environment variables. Please set WHISPER_API_KEY in your .env file."
        )
    return api_key

def get_assisto_api_key(reload: bool = False) -> str:
    """
    Get Assisto API key from environment variables.
    """
    api_key = get_env_var("ASSISTO_API_KEY", reload=reload)
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="ASSISTO_API_KEY not found in environment variables. Please set ASSISTO_API_KEY in your .env file."
        )
    return api_key

def get_current_stt_model() -> str:
    """
    Get the currently selected STT model.
    """
    return current_stt_model

def set_stt_model(model: str) -> bool:
    """
    Set the STT model. Valid models are 'kibo', 'whisper', and 'assisto'.
    """
    global current_stt_model
    if model in ["kibo", "whisper", "assisto"]:
        current_stt_model = model
        logger.info(f"STT model switched to: {model}")
        return True
    else:
        logger.error(f"Invalid STT model: {model}. Valid models are 'kibo', 'whisper', and 'assisto'")
        return False

def ensure_collection():
    """Ensure ChromaDB collection exists."""
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION}' already exists")
    except Exception:
        logger.info(f"Creating ChromaDB collection '{CHROMA_COLLECTION}'")
        collection = chroma_client.create_collection(name=CHROMA_COLLECTION)
        logger.info(f"Successfully created ChromaDB collection '{CHROMA_COLLECTION}'")

def connect_chroma():
    """Connect to ChromaDB."""
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        return client
    except Exception as e:
        logger.error(f"ChromaDB connection failed: {e}")
        return None

def convert_wav_to_mp3(audio_content: bytes, filename: str) -> bytes:
    """
    Convert WAV audio content to MP3 format using pydub.
    """
    try:
        # Load audio from bytes
        audio = AudioSegment.from_wav(BytesIO(audio_content))
        
        # Convert to MP3
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3")
        mp3_content = mp3_buffer.getvalue()
        
        logger.info(f"Successfully converted WAV to MP3: {len(audio_content)} bytes -> {len(mp3_content)} bytes")
        return mp3_content
        
    except Exception as e:
        logger.error(f"Error converting WAV to MP3: {e}")
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")

@retry_with_backoff(max_retries=5, base_delay=3, max_delay=60)
def transcribe_audio_file(audio_file: UploadFile) -> str:
    """
    Transcribe audio file using Kibo AI Speech Transcription API with enhanced error handling and retries.
    """
    try:
        # Check file size (limit to 50MB for better timeout handling)
        max_size = 50 * 1024 * 1024  # 50MB
        content = audio_file.file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")
        
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large ({file_size_mb:.2f} MB). Please upload files smaller than 50MB to avoid timeout issues."
            )
        
        # Convert WAV to MP3 if needed (Kibo AI only supports MP3)
        original_filename = audio_file.filename
        if original_filename.lower().endswith('.wav'):
            logger.info(f"Converting WAV file to MP3 for Kibo AI: {original_filename}")
            content = convert_wav_to_mp3(content, original_filename)
            # Update filename for API call
            audio_file.filename = Path(original_filename).stem + '.mp3'
            logger.info(f"Converted filename: {audio_file.filename}")
        
        # Estimate processing time and warn if file is large
        if file_size_mb > 10:
            logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Processing may take longer.")
        elif file_size_mb > 25:
            logger.warning(f"Very large file detected ({file_size_mb:.2f} MB). Consider using a smaller file if timeout occurs.")
        
        # Get API key (with reload to get latest value)
        api_key = get_kibo_api_key(reload=True)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Kibo AI API configuration
            url = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
            headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            logger.info(f"Sending audio file to Kibo AI API: {audio_file.filename}")
            logger.info(f"API Key configured: {api_key[:8]}..." if len(api_key) > 8 else "Short key")
            logger.info("Starting transcription request... This may take several minutes for large files.")
            
            # Use dynamic timeout based on file size
            timeout_seconds = 300  # 5 minutes base timeout
            if file_size_mb > 10:
                timeout_seconds = 600  # 10 minutes for large files
            elif file_size_mb > 25:
                timeout_seconds = 900  # 15 minutes for very large files
            
            logger.info(f"Using timeout of {timeout_seconds} seconds for {file_size_mb:.2f} MB file")
            
            # Prepare file for upload
            with open(tmp_file_path, "rb") as file:
                files = {
                    "file": (audio_file.filename, file, "audio/mpeg")
                }
                params = {
                    "language_code": "en-IN"  # Use params instead of data for form parameters
                }
                
                try:
                    response = requests.post(url, headers=headers, files=files, params=params, timeout=timeout_seconds)
                    
                    logger.info(f"Kibo AI Response Status: {response.status_code}")
                    logger.info(f"Kibo AI Response Headers: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            logger.info(f"Kibo AI Response JSON: {result}")
                            logger.info(f"Response type: {type(result)}")
                            
                            # Log all keys in the response for debugging
                            if isinstance(result, dict):
                                logger.info(f"Response keys: {list(result.keys())}")
                            
                            # Check for Kibo AI specific response structure
                            transcription = None
                            
                            # Kibo AI returns: {"request_id": "...", "response": [{"transcript": "text", "speaker_id": 0, ...}, ...]}
                            if isinstance(result, dict) and 'response' in result:
                                response_data = result.get('response')
                                if isinstance(response_data, list) and len(response_data) > 0:
                                    # Check if speaker information is available
                                    has_speaker_info = any(
                                        isinstance(segment, dict) and 'speaker_id' in segment 
                                        for segment in response_data
                                    )
                                    
                                    if has_speaker_info:
                                        # Format with speaker information
                                        formatted_segments = []
                                        for segment in response_data:
                                            if isinstance(segment, dict) and 'transcript' in segment:
                                                speaker_id = segment.get('speaker_id', 0)
                                                transcript_text = segment.get('transcript', '').strip()
                                                if transcript_text:
                                                    # Map speaker_id: 0 -> Speaker 1, 1 -> Speaker 2
                                                    speaker_name = "Speaker 1" if speaker_id == 0 else "Speaker 2"
                                                    
                                                    # Split transcript into sentences and format each one separately
                                                    # Use a more sophisticated approach to handle abbreviations and multiple sentences
                                                    sentences = []
                                                    current_sentence = ""
                                                    
                                                    # Split by periods but be careful with abbreviations
                                                    parts = transcript_text.split('.')
                                                    for i, part in enumerate(parts):
                                                        part = part.strip()
                                                        if part:
                                                            # Check if this might be an abbreviation (short part, no space)
                                                            if len(part) <= 3 and ' ' not in part and i < len(parts) - 1:
                                                                # Likely an abbreviation, don't split here
                                                                current_sentence += part + "."
                                                            else:
                                                                # Complete sentence
                                                                if current_sentence:
                                                                    current_sentence += part + "."
                                                                else:
                                                                    current_sentence = part + "."
                                                                
                                                                if current_sentence.strip():
                                                                    sentences.append(current_sentence.strip())
                                                                    current_sentence = ""
                                    
                                                    # Add any remaining sentence
                                                    if current_sentence.strip():
                                                        sentences.append(current_sentence.strip())
                                                    
                                                    # Format each sentence with speaker
                                                    for sentence in sentences:
                                                        if sentence:
                                                            formatted_segments.append(f"{speaker_name}: {sentence}")
                                        
                                        if formatted_segments:
                                            transcription = '\n'.join(formatted_segments)
                                            logger.info(f"Formatted {len(formatted_segments)} individual speaker turns with speaker information")
                                            logger.info(f"Full transcription with speakers (one per line): {transcription[:200]}...")
                                    else:
                                        # Fallback to original format if no speaker info
                                        transcript_segments = []
                                        for segment in response_data:
                                            if isinstance(segment, dict) and 'transcript' in segment:
                                                transcript_text = segment.get('transcript', '').strip()
                                                if transcript_text:
                                                    transcript_segments.append(transcript_text)
                                        
                                        if transcript_segments:
                                            transcription = ' '.join(transcript_segments)
                                            logger.info(f"Combined {len(transcript_segments)} transcript segments (no speaker info)")
                                            logger.info(f"Full transcription: {transcription[:200]}...")
                            
                            # Fallback: Check other possible response structures
                            if transcription is None:
                                possible_keys = ['transcription', 'text', 'transcript', 'result', 'output', 'content', 'data']
                                
                                for key in possible_keys:
                                    if isinstance(result, dict) and key in result:
                                        transcription = result.get(key)
                                        logger.info(f"Found transcription in key '{key}': {transcription}")
                                        break
                            
                            # If still no transcription found, check if result is a string directly
                            if transcription is None:
                                if isinstance(result, str):
                                    transcription = result
                                    logger.info(f"Response is direct string: {transcription}")
                                elif isinstance(result, dict):
                                    # Check if there's any string value in the response (excluding request_id)
                                    for key, value in result.items():
                                        if key != 'request_id' and isinstance(value, str) and value.strip():
                                            transcription = value
                                            logger.info(f"Found string value in key '{key}': {transcription}")
                                            break
                            
                            if transcription and str(transcription).strip():
                                logger.info("Transcription completed successfully")
                                return str(transcription).strip()
                            else:
                                # Return a fallback response instead of throwing error
                                logger.warning(f"No valid transcription found in response. Using fallback.")
                                fallback_message = "Audio transcription completed but content is not available in expected format."
                                
                                # Log the full response for debugging
                                logger.error(f"Full response for debugging: {result}")
                                
                                # If there's any content in the response, try to extract it
                                if isinstance(result, dict) and result:
                                    # Look for any non-empty values
                                    for key, value in result.items():
                                        if value and str(value).strip():
                                            logger.info(f"Using value from '{key}' as fallback: {value}")
                                            return f"[Transcription may be incomplete] {str(value).strip()}"
                                
                                # Return a descriptive message instead of error
                                return f"[Audio processed but transcription format unexpected] Response: {str(result)[:200]}..."
                        except ValueError as json_error:
                            logger.error(f"JSON parsing error: {json_error}")
                            logger.error(f"Raw response: {response.text}")
                            
                            # If JSON parsing fails, try to use the raw text response
                            raw_text = response.text.strip()
                            if raw_text:
                                logger.info(f"Using raw text response as transcription: {raw_text}")
                                return f"[Raw response] {raw_text}"
                            else:
                                raise APIError(f"Invalid JSON response from Kibo AI API and no readable text: {response.text}", status_code=500)
                    else:
                        logger.error(f"Kibo AI API error: {response.status_code}")
                        logger.error(f"Response content: {response.text}")
                        logger.error(f"Response headers: {dict(response.headers)}")
                        
                        # Handle specific error status codes
                        if response.status_code == 401:
                            raise APIError("Unauthorized: Invalid API key for Kibo AI. Please check your API_KEY in .env file.", status_code=401)
                        elif response.status_code == 403:
                            raise APIError("Forbidden: API key doesn't have permission for Kibo AI transcription service.", status_code=403)
                        elif response.status_code == 400:
                            raise APIError(f"Bad Request to Kibo AI API: {response.text}", status_code=400)
                        elif response.status_code == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', 60))
                            raise APIError(f"Rate limited by Kibo AI API. Please wait {retry_after} seconds before retrying.", status_code=429, retry_after=retry_after)
                        elif response.status_code == 503:  # Service unavailable
                            raise APIError("Kibo AI service is temporarily unavailable. Please try again later.", status_code=503)
                        else:
                            raise APIError(f"Kibo AI API error ({response.status_code}): {response.text}", status_code=response.status_code)
                
                except requests.exceptions.Timeout:
                    raise APIError(f"Kibo AI API request timed out after {timeout_seconds} seconds. The audio file might be too large or the service is busy.", status_code=504)
                except requests.exceptions.ConnectionError:
                    raise APIError("Connection error: Unable to connect to Kibo AI API. Please check your internet connection and try again.", status_code=503)
                except requests.exceptions.SSLError:
                    raise APIError("SSL certificate error connecting to Kibo AI API. Please check your system's SSL configuration.", status_code=502)
                except requests.exceptions.RequestException as e:
                    raise APIError(f"Network error connecting to Kibo AI API: {str(e)}", status_code=500)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    except HTTPException:
        raise
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Critical error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Critical error in transcription: {str(e)}")

def transcribe_audio_file_whisper(audio_file: UploadFile) -> str:
    """
    Transcribe audio file using Whisper API via Groq with enhanced error handling.
    """
    try:
        # Check file size (limit to 50MB for better timeout handling)
        max_size = 50 * 1024 * 1024  # 50MB
        content = audio_file.file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")
        
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large ({file_size_mb:.2f} MB). Please upload files smaller than 50MB to avoid timeout issues."
            )
        
        # Convert WAV to MP3 if needed (Whisper supports both but MP3 is more efficient)
        original_filename = audio_file.filename
        if original_filename.lower().endswith('.wav'):
            logger.info(f"Converting WAV file to MP3 for Whisper: {original_filename}")
            content = convert_wav_to_mp3(content, original_filename)
            # Update filename for API call
            audio_file.filename = Path(original_filename).stem + '.mp3'
            logger.info(f"Converted filename: {audio_file.filename}")
        
        # Estimate processing time and warn if file is large
        if file_size_mb > 10:
            logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Processing may take longer.")
        elif file_size_mb > 25:
            logger.warning(f"Very large file detected ({file_size_mb:.2f} MB). Consider using a smaller file if timeout occurs.")
        
        # Get API key (with reload to get latest value)
        api_key = get_whisper_api_key(reload=True)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            logger.info(f"Sending audio file to Whisper API via Groq: {audio_file.filename}")
            logger.info(f"API Key configured: {api_key[:8]}..." if len(api_key) > 8 else "Short key")
            logger.info("Starting Whisper transcription request... This may take several minutes for large files.")
            
            # Initialize Groq client
            client = Groq(api_key=api_key)
            
            # Use dynamic timeout based on file size
            timeout_seconds = 300  # 5 minutes base timeout
            if file_size_mb > 10:
                timeout_seconds = 600  # 10 minutes for large files
            elif file_size_mb > 25:
                timeout_seconds = 900  # 15 minutes for very large files
            
            logger.info(f"Using timeout of {timeout_seconds} seconds for {file_size_mb:.2f} MB file")
            
            # Prepare file for Whisper API
            with open(tmp_file_path, "rb") as file:
                try:
                    transcription = client.audio.transcriptions.create(
                        file=(audio_file.filename, file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )
                    
                    logger.info(f"Whisper API Response: {transcription}")
                    
                    # Extract transcription text
                    if hasattr(transcription, 'text') and transcription.text:
                        transcription_text = transcription.text.strip()
                        logger.info("Whisper transcription completed successfully")
                        logger.info(f"Transcription: {transcription_text[:200]}...")
                        return transcription_text
                    else:
                        logger.warning("No transcription text found in Whisper response")
                        return "Audio transcription completed but no text content was returned."
                        
                except Exception as e:
                    logger.error(f"Whisper API error: {str(e)}")
                    raise APIError(f"Whisper API error: {str(e)}", status_code=500)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    except HTTPException:
        raise
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Critical error in Whisper transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Critical error in Whisper transcription: {str(e)}")

@retry_with_backoff(max_retries=5, base_delay=3, max_delay=60)
def transcribe_audio_file_assisto(audio_file: UploadFile) -> str:
    """
    Transcribe audio file using Assisto API with enhanced error handling and retries.
    """
    try:
        # Check file size (limit to 50MB for better timeout handling)
        max_size = 50 * 1024 * 1024  # 50MB
        content = audio_file.file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        logger.info(f"Audio file size: {file_size_mb:.2f} MB")
        
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large ({file_size_mb:.2f} MB). Please upload files smaller than 50MB to avoid timeout issues."
            )
        
        # Assisto API supports both MP3 and WAV files
        original_filename = audio_file.filename
        logger.info(f"Processing audio file: {original_filename}")
        
        # Estimate processing time and warn if file is large
        if file_size_mb > 10:
            logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Processing may take longer.")
        elif file_size_mb > 25:
            logger.warning(f"Very large file detected ({file_size_mb:.2f} MB). Consider using a smaller file if timeout occurs.")
        
        # Get API key (with reload to get latest value)
        api_key = get_assisto_api_key(reload=True)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Assisto API configuration
            url = "https://dev.assisto.tech/workflow_apis/process_file"
            
            payload = {'workflow_name': 'diarization'}
            headers = {
                'Authorization': f'Bearer {api_key}'
            }
            
            logger.info(f"Sending audio file to Assisto API: {audio_file.filename}")
            logger.info(f"API Key configured: {api_key[:8]}..." if len(api_key) > 8 else "Short key")
            logger.info("Starting transcription request... This may take several minutes for large files.")
            
            # Use dynamic timeout based on file size
            timeout_seconds = 300  # 5 minutes base timeout
            if file_size_mb > 10:
                timeout_seconds = 600  # 10 minutes for large files
            elif file_size_mb > 25:
                timeout_seconds = 900  # 15 minutes for very large files
            
            logger.info(f"Using timeout of {timeout_seconds} seconds for {file_size_mb:.2f} MB file")
            
            # Prepare file for upload
            with open(tmp_file_path, "rb") as file:
                files = [
                    ('file', (audio_file.filename, file, 'application/octet-stream'))
                ]
                
                try:
                    response = requests.post(url, headers=headers, data=payload, files=files, timeout=timeout_seconds)
                    
                    logger.info(f"Assisto API Response Status: {response.status_code}")
                    logger.info(f"Assisto API Response Headers: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            logger.info(f"Assisto API Response JSON: {result}")
                            logger.info(f"Response type: {type(result)}")
                            
                            # Log all keys in the response for debugging
                            if isinstance(result, dict):
                                logger.info(f"Response keys: {list(result.keys())}")
                            
                            # Parse Assisto API response
                            transcription = None
                            
                            # Try to extract transcription from various possible keys
                            possible_keys = ['transcription', 'text', 'transcript', 'result', 'output', 'content', 'data', 'response']
                            
                            for key in possible_keys:
                                if isinstance(result, dict) and key in result:
                                    transcription = result.get(key)
                                    logger.info(f"Found transcription in key '{key}': {transcription}")
                                    break
                            
                            # If still no transcription found, check if result is a string directly
                            if transcription is None:
                                if isinstance(result, str):
                                    transcription = result
                                    logger.info(f"Response is direct string: {transcription}")
                                elif isinstance(result, dict):
                                    # Check if there's any string value in the response
                                    for key, value in result.items():
                                        if isinstance(value, str) and value.strip():
                                            transcription = value
                                            logger.info(f"Found string value in key '{key}': {transcription}")
                                            break
                            
                            if transcription and str(transcription).strip():
                                logger.info("Assisto transcription completed successfully")
                                return str(transcription).strip()
                            else:
                                # Return a fallback response instead of throwing error
                                logger.warning(f"No valid transcription found in response. Using fallback.")
                                fallback_message = "Audio transcription completed but content is not available in expected format."
                                
                                # Log the full response for debugging
                                logger.error(f"Full response for debugging: {result}")
                                
                                # If there's any content in the response, try to extract it
                                if isinstance(result, dict) and result:
                                    # Look for any non-empty values
                                    for key, value in result.items():
                                        if value and str(value).strip():
                                            logger.info(f"Using value from '{key}' as fallback: {value}")
                                            return f"[Transcription may be incomplete] {str(value).strip()}"
                                
                                # Return a descriptive message instead of error
                                return f"[Audio processed but transcription format unexpected] Response: {str(result)[:200]}..."
                        except ValueError as json_error:
                            logger.error(f"JSON parsing error: {json_error}")
                            logger.error(f"Raw response: {response.text}")
                            
                            # If JSON parsing fails, try to use the raw text response
                            raw_text = response.text.strip()
                            if raw_text:
                                logger.info(f"Using raw text response as transcription: {raw_text}")
                                return f"[Raw response] {raw_text}"
                            else:
                                raise APIError(f"Invalid JSON response from Assisto API and no readable text: {response.text}", status_code=500)
                    else:
                        logger.error(f"Assisto API error: {response.status_code}")
                        logger.error(f"Response content: {response.text}")
                        logger.error(f"Response headers: {dict(response.headers)}")
                        
                        # Handle specific error status codes
                        if response.status_code == 401:
                            raise APIError("Unauthorized: Invalid API key for Assisto API. Please check your ASSISTO_API_KEY in .env file.", status_code=401)
                        elif response.status_code == 403:
                            raise APIError("Forbidden: API key doesn't have permission for Assisto transcription service.", status_code=403)
                        elif response.status_code == 400:
                            raise APIError(f"Bad Request to Assisto API: {response.text}", status_code=400)
                        elif response.status_code == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', 60))
                            raise APIError(f"Rate limited by Assisto API. Please wait {retry_after} seconds before retrying.", status_code=429, retry_after=retry_after)
                        elif response.status_code == 503:  # Service unavailable
                            raise APIError("Assisto service is temporarily unavailable. Please try again later.", status_code=503)
                        else:
                            raise APIError(f"Assisto API error ({response.status_code}): {response.text}", status_code=response.status_code)
                
                except requests.exceptions.Timeout:
                    raise APIError(f"Assisto API request timed out after {timeout_seconds} seconds. The audio file might be too large or the service is busy.", status_code=504)
                except requests.exceptions.ConnectionError:
                    raise APIError("Connection error: Unable to connect to Assisto API. Please check your internet connection and try again.", status_code=503)
                except requests.exceptions.SSLError:
                    raise APIError("SSL certificate error connecting to Assisto API. Please check your system's SSL configuration.", status_code=502)
                except requests.exceptions.RequestException as e:
                    raise APIError(f"Network error connecting to Assisto API: {str(e)}", status_code=500)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    except HTTPException:
        raise
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Critical error in Assisto transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Critical error in Assisto transcription: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "GST & License Analyzer API is running", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Legacy health check endpoint for backward compatibility."""
    return {"status": "healthy", "service": "tally-poc-backend"}

@app.get("/health")
async def detailed_health_check():
    """Detailed health check endpoint."""
    kibo_api_key = get_env_var("API_KEY", reload=True)
    watsonx_api_key = get_env_var("WATSONX_API_KEY", reload=True)
    project_id = get_env_var("WATSONX_PROJECT_ID", reload=True)
    model_id = get_env_var("WATSONX_MODEL_ID", reload=True)
    whisper_api_key = get_env_var("WHISPER_API_KEY", reload=True)
    assisto_api_key = get_env_var("ASSISTO_API_KEY", reload=True)
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "environment": {
            "kibo_api_key_configured": bool(kibo_api_key),
            "watsonx_api_key_configured": bool(watsonx_api_key),
            "project_id_configured": bool(project_id),
            "model_id_configured": bool(model_id),
            "whisper_api_key_configured": bool(whisper_api_key),
            "assisto_api_key_configured": bool(assisto_api_key),
        },
        "token_status": {
            "has_cached_token": bool(access_token_cache["token"]),
            "token_valid": time.time() < access_token_cache["expiry"] if access_token_cache["token"] else False
        },
        "services": {},
        "stt_model": {
            "current_model": get_current_stt_model(),
            "available_models": ["kibo", "whisper", "assisto"]
        }
    }
    
    # Test Kibo AI service
    try:
        if kibo_api_key:
            kibo_status = await check_kibo_status()
            health_status["services"]["kibo_ai"] = {
                "status": "healthy" if kibo_status.get("api_responsive") else "unhealthy",
                "response_time": kibo_status.get("response_time_seconds"),
                "details": kibo_status
            }
        else:
            health_status["services"]["kibo_ai"] = {
                "status": "not_configured",
                "message": "API key not configured"
            }
    except Exception as e:
        health_status["services"]["kibo_ai"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test WatsonX service
    try:
        if watsonx_api_key and project_id:
            watson_status = await test_watson_credentials()
            health_status["services"]["watsonx"] = {
                "status": "healthy" if watson_status.get("valid") else "unhealthy",
                "details": watson_status
            }
        else:
            health_status["services"]["watsonx"] = {
                "status": "not_configured",
                "message": "API key or project ID not configured"
            }
    except Exception as e:
        health_status["services"]["watsonx"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test Assisto AI service
    try:
        if assisto_api_key:
            assisto_status = await check_assisto_status()
            health_status["services"]["assisto_ai"] = {
                "status": "healthy" if assisto_status.get("api_responsive") else "unhealthy",
                "response_time": assisto_status.get("response_time_seconds"),
                "details": assisto_status
            }
        else:
            health_status["services"]["assisto_ai"] = {
                "status": "not_configured",
                "message": "API key not configured"
            }
    except Exception as e:
        health_status["services"]["assisto_ai"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test ChromaDB service
    try:
        chroma_status = test_chroma_connection()
        health_status["services"]["chroma"] = {
            "status": "healthy" if chroma_status.get("connected") else "unhealthy",
            "details": chroma_status
        }
    except Exception as e:
        health_status["services"]["chroma"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Determine overall health status
    all_services_healthy = all(
        service.get("status") == "healthy" 
        for service in health_status["services"].values()
    )
    
    if not all_services_healthy:
        health_status["status"] = "degraded"
        health_status["message"] = "Some services are experiencing issues"
    
    return health_status

@app.get("/health/simple")
async def simple_health_check():
    """Simple health check for load balancers."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/health/services")
async def services_health_check():
    """Check health of individual services."""
    services = {}
    
    # Kibo AI Health Check
    try:
        kibo_status = await check_kibo_status()
        services["kibo_ai"] = {
            "healthy": kibo_status.get("api_responsive", False),
            "response_time": kibo_status.get("response_time_seconds"),
            "status_code": kibo_status.get("status_code")
        }
    except Exception as e:
        services["kibo_ai"] = {"healthy": False, "error": str(e)}
    
    # WatsonX Health Check
    try:
        watson_status = await test_watson_credentials()
        services["watsonx"] = {
            "healthy": watson_status.get("valid", False),
            "message": watson_status.get("message")
        }
    except Exception as e:
        services["watsonx"] = {"healthy": False, "error": str(e)}
    
    # Assisto AI Health Check
    try:
        assisto_status = await check_assisto_status()
        services["assisto_ai"] = {
            "healthy": assisto_status.get("api_responsive", False),
            "response_time": assisto_status.get("response_time_seconds"),
            "status_code": assisto_status.get("status_code")
        }
    except Exception as e:
        services["assisto_ai"] = {"healthy": False, "error": str(e)}
    
    # ChromaDB Health Check
    try:
        chroma_status = test_chroma_connection()
        services["chroma"] = {
            "healthy": chroma_status.get("connected", False),
            "message": chroma_status.get("message")
        }
    except Exception as e:
        services["chroma"] = {"healthy": False, "error": str(e)}
    
    return {
        "timestamp": time.time(),
        "services": services,
        "overall_healthy": all(service.get("healthy", False) for service in services.values())
    }

@app.get("/test-connectivity")
async def test_all_services_connectivity():
    """Test connectivity to all external services."""
    results = {}
    
    # Test Kibo AI connectivity
    try:
        kibo_url = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
        api_key = os.getenv("API_KEY")
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        kibo_result = await test_service_connectivity(kibo_url, headers, timeout=15)
        results["kibo_ai"] = {
            "url": kibo_url,
            "status": "connected",
            "details": kibo_result
        }
    except APIError as e:
        results["kibo_ai"] = {
            "url": kibo_url,
            "status": "failed",
            "error": str(e),
            "status_code": e.status_code
        }
    
    # Test WatsonX connectivity
    try:
        watsonx_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat"
        watsonx_result = await test_service_connectivity(watsonx_url, timeout=15)
        results["watsonx"] = {
            "url": watsonx_url,
            "status": "connected",
            "details": watsonx_result
        }
    except APIError as e:
        results["watsonx"] = {
            "url": watsonx_url,
            "status": "failed",
            "error": str(e),
            "status_code": e.status_code
        }
    
    # Test IBM Cloud IAM connectivity
    try:
        iam_url = "https://iam.cloud.ibm.com/identity/token"
        iam_result = await test_service_connectivity(iam_url, timeout=15)
        results["ibm_iam"] = {
            "url": iam_url,
            "status": "connected",
            "details": iam_result
        }
    except APIError as e:
        results["ibm_iam"] = {
            "url": iam_url,
            "status": "failed",
            "error": str(e),
            "status_code": e.status_code
        }
    
    return {
        "timestamp": time.time(),
        "connectivity_tests": results,
        "summary": {
            "total_services": len(results),
            "connected": len([r for r in results.values() if r.get("status") == "connected"]),
            "failed": len([r for r in results.values() if r.get("status") == "failed"])
        }
    }

# Enhanced error handling for ChromaDB operations
@retry_with_backoff(max_retries=5, base_delay=2, max_delay=30)
def safe_chroma_operation(operation_name: str, operation_func, *args, **kwargs):
    """Safely execute ChromaDB operations with retry logic."""
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"ChromaDB {operation_name} failed: {str(e)}")
        raise APIError(f"ChromaDB {operation_name} failed: {str(e)}", status_code=500)

# Enhanced error handling for audio file processing
@app.post("/api/upload/audio")
async def upload_audio(files: list[UploadFile] = File(...), tag: str = "default"):
    """Upload audio files to ChromaDB database."""
    saved_files, points = [], []
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        
        # Rename WAV files to MP3 for ChromaDB storage (name only, no conversion)
        original_filename = file.filename
        if ext == '.wav':
            # Change extension from .wav to .mp3 for storage
            storage_filename = Path(file.filename).stem + '.mp3'
            logger.info(f"Renaming WAV file: {original_filename} -> {storage_filename}")
        else:
            storage_filename = original_filename
        
        logger.info(f"Processing file: {original_filename} ({size_mb:.2f} MB)")
        
        if size_mb > MAX_FILE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.2f} MB (max {MAX_FILE_MB} MB)",
            )

        saved_files.append(original_filename)

        # Encode audio as base64 for storage
        b64_audio = base64.b64encode(content).decode("utf-8")
        text = f"Audio uploaded: {storage_filename}. Transcript pending."

        # Create document for ChromaDB
        doc_id = uuid4().hex
        points.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "source": storage_filename,  # Use renamed filename for storage
                "original_filename": original_filename,  # Keep original name for reference
                "tag": tag,
                "type": "audio",
                "audio_base64": b64_audio,
                "file_size_mb": size_mb,
                "upload_timestamp": time.time()
            }
        })

        await file.close()

    # Upload documents to ChromaDB with retry logic
    if points:
        try:
            safe_chroma_operation("add", collection.add, 
                ids=[p["id"] for p in points],
                documents=[p["text"] for p in points],
                metadatas=[p["metadata"] for p in points]
            )
            logger.info(f"Successfully uploaded {len(points)} files to ChromaDB")
        except APIError as api_err:
            logger.error(f"Error uploading to ChromaDB: {api_err}")
            raise HTTPException(status_code=500, detail=f"Error uploading to ChromaDB: {str(api_err)}")

    return {
        "message": "Upload completed successfully", 
        "files": saved_files,
        "uploaded_to_chroma": len(points)
    }

@app.post("/api/search")
async def search_audio(body: SearchIn):
    """Search audio files in ChromaDB database."""
    try:
        # Build where clause if tag is specified
        where_clause = None
        if body.tag:
            where_clause = {"tag": body.tag}
        
        # Search in ChromaDB with retry logic
        try:
            res = safe_chroma_operation("query", collection.query,
                query_texts=[body.query],
                n_results=body.top_k,
                where=where_clause
            )
        except APIError as api_err:
            logger.error(f"ChromaDB search failed: {api_err}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(api_err)}")
        
        matches = []
        if res and res['ids'] and res['ids'][0]:
            for i, doc_id in enumerate(res['ids'][0]):
                match_data = {
                    "id": doc_id,
                    "score": res['distances'][0][i] if res['distances'] and res['distances'][0] else 0,
                    "source": res['metadatas'][0][i].get("source") if res['metadatas'] and res['metadatas'][0] else None,
                    "text": res['documents'][0][i] if res['documents'] and res['documents'][0] else None,
                    "tag": res['metadatas'][0][i].get("tag") if res['metadatas'] and res['metadatas'][0] else None,
                    "type": res['metadatas'][0][i].get("type") if res['metadatas'] and res['metadatas'][0] else None,
                    "file_size_mb": res['metadatas'][0][i].get("file_size_mb") if res['metadatas'] and res['metadatas'][0] else None,
                    "upload_timestamp": res['metadatas'][0][i].get("upload_timestamp") if res['metadatas'] and res['metadatas'][0] else None
                }
                
                # Only include audio_base64 if explicitly requested (it's large)
                if body.query.lower().find("download") != -1 or body.query.lower().find("audio") != -1:
                    match_data["audio_base64"] = res['metadatas'][0][i].get("audio_base64") if res['metadatas'] and res['metadatas'][0] else None
                
                matches.append(match_data)
        
        return {
            "query": body.query,
            "total_matches": len(matches),
            "matches": matches
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@app.get("/collection/all")
def fetch_all_data():
    """Fetch all data from ChromaDB collection."""
    try:
        res = safe_chroma_operation("get", collection.get, limit=1000)
        
        result = {
            "status": "success",
            "collection": CHROMA_COLLECTION,
            "total_points": len(res['ids']) if res['ids'] else 0,
            "data": [],
        }

        if res['ids']:
            for i, doc_id in enumerate(res['ids']):
                item = {
                    "id": doc_id,
                    "payload": {k: v for k, v in res['metadatas'][i].items() if k != "audio_base64"} if res['metadatas'] and res['metadatas'][i] else {},  # Exclude large audio data
                }
                result["data"].append(item)

        return result
    except APIError as api_err:
        logger.error(f"Error fetching collection data: {api_err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch collection data: {str(api_err)}")
    except Exception as e:
        logger.error(f"Error fetching collection data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch collection data: {e}")

@app.get("/collection/info")
def get_collection_info():
    """Get ChromaDB collection information."""
    try:
        collection_info = safe_chroma_operation("count", collection.count)
        return {
            "collection_name": CHROMA_COLLECTION,
            "status": "active",
            "points_count": collection_info,
            "segments_count": 0,  # ChromaDB doesn't expose this
            "vector_size": 384,  # Default embedding size
            "distance": "cosine"  # Default distance metric
        }
    except APIError as api_err:
        logger.error(f"Error getting collection info: {api_err}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(api_err)}")
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {e}")

@app.get("/collection/audio/{point_id}")
def get_audio_data(point_id: str):
    """Get audio data for a specific point ID."""
    try:
        # Retrieve the specific document by ID with retry logic
        try:
            res = safe_chroma_operation("get", collection.get, ids=[point_id])
        except APIError as api_err:
            logger.error(f"Error retrieving audio data: {api_err}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve audio data: {str(api_err)}")
        
        if not res['ids'] or len(res['ids']) == 0:
            raise HTTPException(status_code=404, detail=f"Audio file with ID {point_id} not found")
        
        metadata = res['metadatas'][0] if res['metadatas'] and res['metadatas'] else {}
        audio_base64 = metadata.get("audio_base64")
        
        if not audio_base64:
            raise HTTPException(status_code=404, detail=f"No audio data found for ID {point_id}")
        
        return {
            "id": point_id,
            "audio_base64": audio_base64,
            "source": metadata.get("source"),
            "original_filename": metadata.get("original_filename"),
            "file_size_mb": metadata.get("file_size_mb"),
            "upload_timestamp": metadata.get("upload_timestamp")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audio data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio data: {e}")

@app.get("/test-chroma-connection")
def test_chroma_connection():
    """Test ChromaDB database connection."""
    try:
        # Test connection
        client = connect_chroma()
        if not client:
            return {
                "connected": False,
                "message": "Failed to connect to ChromaDB",
                "config": {
                    "persist_directory": CHROMA_PERSIST_DIRECTORY,
                    "collection": CHROMA_COLLECTION
                }
            }
        
        # Test collection access with retry logic
        try:
            collection_count = safe_chroma_operation("count", collection.count)
        except APIError as api_err:
            return {
                "connected": False,
                "message": f"ChromaDB connection test failed: {str(api_err)}",
                "config": {
                    "persist_directory": CHROMA_PERSIST_DIRECTORY,
                    "collection": CHROMA_COLLECTION
                }
            }
        
        return {
            "connected": True,
            "message": "Successfully connected to ChromaDB",
            "config": {
                "persist_directory": CHROMA_PERSIST_DIRECTORY,
                "collection": CHROMA_COLLECTION
            },
            "collection_info": {
                "status": "active",
                "points_count": collection_count,
                "vector_size": 384
            }
        }
    except Exception as e:
        logger.error(f"ChromaDB connection test failed: {e}")
        return {
            "connected": False,
            "message": f"ChromaDB connection test failed: {e}",
            "config": {
                "persist_directory": CHROMA_PERSIST_DIRECTORY,
                "collection": CHROMA_COLLECTION
            }
        }

@app.post("/refresh-env")
async def refresh_environment_variables():
    """Manually refresh environment variables from .env file and clear caches."""
    try:
        # Reload environment variables
        reload_env_variables()
        
        # Clear token cache
        clear_token_cache()
        
        # Get updated values
        watsonx_api_key = get_env_var("WATSONX_API_KEY")
        kibo_api_key = get_env_var("API_KEY")
        project_id = get_env_var("WATSONX_PROJECT_ID")
        model_id = get_env_var("WATSONX_MODEL_ID")
        
        return {
            "success": True,
            "message": "Environment variables refreshed successfully",
            "updated_values": {
                "watsonx_api_key_configured": bool(watsonx_api_key),
                "kibo_api_key_configured": bool(kibo_api_key),
                "project_id_configured": bool(project_id),
                "model_id_configured": bool(model_id),
                "token_cache_cleared": True
            },
            "api_key_previews": {
                "watsonx": f"{watsonx_api_key[:4]}...{watsonx_api_key[-4:]}" if watsonx_api_key and len(watsonx_api_key) >= 8 else "not_configured",
                "kibo": f"{kibo_api_key[:4]}...{kibo_api_key[-4:]}" if kibo_api_key and len(kibo_api_key) >= 8 else "not_configured"
            }
        }
    except Exception as e:
        logger.error(f"Error refreshing environment variables: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to refresh environment variables: {str(e)}"
        }

@app.get("/test-watson-credentials")
async def test_watson_credentials():
    """Test if Watson credentials are valid without making a full request."""
    try:
        # Reload environment variables to get latest values
        api_key = get_env_var("WATSONX_API_KEY", reload=True)
        project_id = get_env_var("WATSONX_PROJECT_ID", reload=True)
        model_id = get_env_var("WATSONX_MODEL_ID", reload=True)
        
        if not api_key or api_key == "your_actual_watson_api_key_here":
            return {
                "valid": False,
                "message": "API key not configured or still using placeholder value",
                "instruction": "Please update WATSONX_API_KEY in your .env file"
            }
        
        if not project_id or project_id == "your_actual_project_id_here":
            return {
                "valid": False,
                "message": "Project ID not configured or still using placeholder value",
                "instruction": "Please update WATSONX_PROJECT_ID in your .env file"
            }
        
        # Test token generation
        try:
            token = get_access_token(api_key)
            if token:
                return {
                    "valid": True,
                    "message": "Watson credentials are valid and working",
                    "api_key_format": f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "short_key",
                    "project_id": project_id,
                    "model_id": model_id
                }
            else:
                return {
                    "valid": False,
                    "message": "Failed to get access token",
                    "instruction": "Check your API key and try again"
                }
        except HTTPException as e:
            return {
                "valid": False,
                "message": f"Credential validation failed: {e.detail}",
                "instruction": "Please check your Watson API key and project ID"
            }
        
    except Exception as e:
        logger.error(f"Credential test error: {str(e)}")
        return {
            "valid": False,
            "message": f"Error testing credentials: {str(e)}",
            "instruction": "Check your .env file configuration"
        }

@app.get("/debug-kibo-api")
async def debug_kibo_api():
    """Debug endpoint to test Kibo API response structure."""
    try:
        api_key = get_kibo_api_key()
        
        url = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # Try different request methods to understand the API
        debug_info = {
            "api_url": url,
            "headers_sent": {"accept": headers["accept"], "auth_header_present": bool(headers.get("Authorization"))},
            "api_key_format": f"{api_key[:8]}..." if len(api_key) > 8 else "short_key"
        }
        
        # Test with GET request (might return API info)
        try:
            get_response = requests.get(url, headers=headers, timeout=10)
            debug_info["get_test"] = {
                "status_code": get_response.status_code,
                "headers": dict(get_response.headers),
                "content": get_response.text[:500] if get_response.text else None
            }
        except Exception as e:
            debug_info["get_test"] = {"error": str(e)}
        
        # Test with POST request (no file, might return format info)
        try:
            post_response = requests.post(url, headers=headers, timeout=10)
            debug_info["post_test"] = {
                "status_code": post_response.status_code,
                "headers": dict(post_response.headers),
                "content": post_response.text[:500] if post_response.text else None
            }
        except Exception as e:
            debug_info["post_test"] = {"error": str(e)}
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Debug test failed: {str(e)}"}

@app.get("/test-kibo-credentials")
async def test_kibo_credentials():
    """Test if Kibo API credentials are valid."""
    try:
        api_key = get_env_var("API_KEY", reload=True)
        
        if not api_key:
            return {
                "valid": False,
                "message": "Kibo API key not configured",
                "instruction": "Please set API_KEY in your .env file"
            }
        
        if api_key == "your_actual_kibo_api_key_here":
            return {
                "valid": False,
                "message": "Kibo API key still using placeholder value",
                "instruction": "Please update API_KEY in your .env file with your actual Kibo API key"
            }
        
        # Test API key format (basic validation)
        if len(api_key) < 10:
            return {
                "valid": False,
                "message": "Kibo API key appears to be too short",
                "instruction": "Please check your API_KEY in the .env file"
            }
        
        # Test API connectivity
        try:
            url = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
            headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            # Make a simple GET request to test connectivity (this might fail but will show response)
            test_response = requests.get(url, headers=headers, timeout=10)
            
            return {
                "valid": True,
                "message": "Kibo API key is configured and API is reachable",
                "api_key_format": f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "short_key",
                "api_test_status": test_response.status_code,
                "instruction": "API connectivity test completed. Upload an audio file to test transcription."
            }
        except requests.exceptions.RequestException as req_error:
            return {
                "valid": True,
                "message": "Kibo API key is configured but connectivity test failed",
                "api_key_format": f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "short_key",
                "connectivity_error": str(req_error),
                "instruction": "API key format looks valid but there may be network issues. Try uploading an audio file."
            }
        
    except Exception as e:
        logger.error(f"Kibo credential test error: {str(e)}")
        return {
            "valid": False,
            "message": f"Error testing Kibo credentials: {str(e)}",
            "instruction": "Check your .env file configuration"
        }

@app.get("/check-kibo-status")
async def check_kibo_status():
    """Quick check if Kibo AI API is responsive."""
    try:
        api_key = get_kibo_api_key()
        url = 'https://api-dev.trestlelabs.com/stt/stt_translate/v1'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # Quick GET request to check if API is responsive
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=10)
        response_time = time.time() - start_time
        
        return {
            "api_responsive": True,
            "response_time_seconds": round(response_time, 2),
            "status_code": response.status_code,
            "message": f"Kibo AI API responded in {response_time:.2f} seconds"
        }
    except requests.exceptions.Timeout:
        return {
            "api_responsive": False,
            "error": "API timeout - service may be slow or unavailable",
            "message": "Consider waiting before uploading large files"
        }
    except Exception as e:
        return {
            "api_responsive": False,
            "error": str(e),
            "message": "API connectivity issue detected"
        }

@app.get("/check-assisto-status")
async def check_assisto_status():
    """Quick check if Assisto API is responsive."""
    try:
        api_key = get_assisto_api_key()
        url = "https://dev.assisto.tech/workflow_apis/process_file"
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        # Quick GET request to check if API is responsive
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=10)
        response_time = time.time() - start_time
        
        return {
            "api_responsive": True,
            "response_time_seconds": round(response_time, 2),
            "status_code": response.status_code,
            "message": f"Assisto API responded in {response_time:.2f} seconds"
        }
    except requests.exceptions.Timeout:
        return {
            "api_responsive": False,
            "error": "API timeout - service may be slow or unavailable",
            "message": "Consider waiting before uploading large files"
        }
    except Exception as e:
        return {
            "api_responsive": False,
            "error": str(e),
            "message": "API connectivity issue detected"
        }

@app.post("/test-transcribe")
async def test_transcribe_only(file: UploadFile = File(...)):
    """Test transcription only (without WatsonX) for debugging."""
    try:
        if not file.filename.lower().endswith('.mp3'):
            raise HTTPException(status_code=400, detail="Only MP3 files are supported")
        
        # Test API key first
        try:
            api_key = get_kibo_api_key()
            logger.info(f"Using API key for test: {api_key[:8]}..." if len(api_key) > 8 else "Short key")
        except HTTPException as key_error:
            raise HTTPException(status_code=500, detail=f"API key error: {key_error.detail}")
        
        transcription = transcribe_audio_file(file)
        
        return {
            "transcription": transcription,
            "success": True,
            "message": "Transcription completed successfully (test mode)",
            "filename": file.filename,
            "api_used": "Kibo AI"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test transcription failed: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Transcribe audio file to text using the currently selected STT model."""
    try:
        if not file.filename.lower().endswith('.mp3'):
            raise HTTPException(status_code=400, detail="Only MP3 files are supported")
        
        # Get current STT model and transcribe accordingly
        current_model = get_current_stt_model()
        logger.info(f"Using STT model: {current_model}")
        
        if current_model == "whisper":
            transcription = transcribe_audio_file_whisper(file)
        elif current_model == "assisto":
            transcription = transcribe_audio_file_assisto(file)
        else:  # Default to kibo
            transcription = transcribe_audio_file(file)
        
        return TranscriptionResponse(
            transcription=transcription,
            success=True,
            message=f"Transcription completed successfully using {current_model} model"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/stt-model")
async def get_stt_model():
    """Get the currently selected STT model."""
    return {
        "model": get_current_stt_model(),
        "available_models": ["kibo", "whisper", "assisto"],
        "success": True
    }

@app.post("/stt-model")
async def set_stt_model_endpoint(model_data: dict):
    """Set the STT model."""
    try:
        model = model_data.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="Model parameter is required")
        
        if set_stt_model(model):
            return {
                "model": get_current_stt_model(),
                "message": f"STT model switched to {model}",
                "success": True
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid model. Valid models are 'kibo', 'whisper', and 'assisto'")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT model switch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to switch STT model: {str(e)}")

@app.post("/recognize-intent", response_model=IntentResponse)
async def recognize_intent_endpoint(content: dict):
    """Recognize intent from transcribed text."""
    try:
        api_key = get_env_var("WATSONX_API_KEY", reload=True)
        project_id = get_env_var("WATSONX_PROJECT_ID", reload=True)
        model_id = get_env_var("WATSONX_MODEL_ID", reload=True)
        
        if not all([api_key, project_id, model_id]):
            raise HTTPException(
                status_code=500, 
                detail="Missing required environment variables (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID)"
            )
        
        text_content = content.get("text", "")
        if not text_content:
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Get intent recognition prompts from separate file
        system_message = get_intent_system_message()
        user_message = get_intent_user_message(text_content)
        
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "parameters": {
    "decoding_method": "greedy",
    "max_new_tokens": 800,
    "temperature": 0.1,        # This might still be too high
    "top_p": 0.9,             # This is definitely too high
    "repetition_penalty": 1.1  # This could cause variation
},

            "model_id": model_id,
            "project_id": project_id
        }
        
        try:
            result = make_watsonx_request(payload, api_key)
            
            if result:
                intent_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return IntentResponse(
                    intent=intent_response.strip(),
                    success=True,
                    message="Intent recognition completed successfully"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to get intent recognition response")
        
        except APIError as api_err:
            logger.error(f"WatsonX API error in intent recognition: {str(api_err)}")
            if api_err.status_code == 504:
                raise HTTPException(status_code=504, detail="Intent recognition timed out. Please try again.")
            elif api_err.status_code == 503:
                raise HTTPException(status_code=503, detail="WatsonX service is temporarily unavailable. Please try again later.")
            elif api_err.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limited. Please wait before trying again.")
            else:
                raise HTTPException(status_code=500, detail=f"Intent recognition failed: {str(api_err)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intent recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Intent recognition failed: {str(e)}")

@app.post("/extract-keywords", response_model=KeywordExtractionResponse)
async def extract_keywords_endpoint(content: dict):
    """Extract keywords and entities from transcribed text."""
    try:
        api_key = get_env_var("WATSONX_API_KEY", reload=True)
        project_id = get_env_var("WATSONX_PROJECT_ID", reload=True)
        model_id = get_env_var("WATSONX_MODEL_ID", reload=True)
        
        if not all([api_key, project_id, model_id]):
            raise HTTPException(
                status_code=500, 
                detail="Missing required environment variables (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID)"
            )
        
        text_content = content.get("text", "")
        if not text_content:
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Get keyword extraction prompts
        system_message = get_keyword_extraction_system_message()
        user_message = get_keyword_extraction_user_message(text_content)
        
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "model_id": model_id,
            "project_id": project_id
        }
        
        try:
            result = make_watsonx_request(payload, api_key)
            
            if result:
                raw_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Validate and parse the extraction response
                extraction_result = validate_extraction_response(raw_response)
                
                return KeywordExtractionResponse(
                    keywords=extraction_result.get("keywords", []),
                    overall_confidence=extraction_result.get("overall_confidence", 50),
                    success=True,
                    message="Keyword extraction completed successfully"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to get keyword extraction response")
        
        except APIError as api_err:
            logger.error(f"WatsonX API error in keyword extraction: {str(api_err)}")
            if api_err.status_code == 504:
                raise HTTPException(status_code=504, detail="Keyword extraction timed out. Please try again.")
            elif api_err.status_code == 503:
                raise HTTPException(status_code=503, detail="WatsonX service is temporarily unavailable. Please try again later.")
            elif api_err.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limited. Please wait before trying again.")
            else:
                # Fallback to simple extraction if WatsonX fails
                logger.info("Using fallback keyword extraction due to API error")
                fallback_result = extract_keywords_simple_fallback(text_content)
                return KeywordExtractionResponse(
                    keywords=fallback_result.get("keywords", []),
                    overall_confidence=fallback_result.get("overall_confidence", 50),
                    success=True,
                    message="Keyword extraction completed using fallback method"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Keyword extraction error: {str(e)}")
        # Use fallback extraction for any other errors
        try:
            text_content = content.get("text", "")
            fallback_result = extract_keywords_simple_fallback(text_content)
            return KeywordExtractionResponse(
                keywords=fallback_result.get("keywords", []),
                overall_confidence=fallback_result.get("overall_confidence", 50),
                success=True,
                message="Keyword extraction completed using fallback method"
            )
        except Exception as fallback_error:
            logger.error(f"Fallback keyword extraction failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")

# Add this import at the top (replace your existing rag import)
from rag_pipeline import knowledge_retriever

# Update your existing /generate-resolution endpoint
@app.post("/generate-resolution", response_model=ResolutionResponse)
async def generate_resolution_endpoint(data: dict):
    """Generate resolution with RAG enhancement."""
    try:
        api_key = get_env_var("WATSONX_API_KEY", reload=True)
        project_id = get_env_var("WATSONX_PROJECT_ID", reload=True)
        model_id = get_env_var("WATSONX_MODEL_ID", reload=True)

        if not all([api_key, project_id, model_id]):
            raise HTTPException(status_code=500, detail="Missing required environment variables")

        content = data.get("content", "")
        intent = data.get("intent", "")
        filename = data.get("filename", "audio_file")
        
        if not content or not intent:
            raise HTTPException(status_code=400, detail="Both content and intent are required")

        # RAG Enhancement: Retrieve context
        logger.info("Retrieving context using RAG...")
        rag_context = knowledge_retriever.retrieve_context(content)
        
        # Build prompt with RAG context
        prompt = f"""
Based on the transcript and knowledge base context, provide a structured response:

File: {filename}
Identified Intent: {intent}
Recommended Resolution Path:
1. [Steps based on context below]
Suggested Knowledge Base Article:
{rag_context.get('chosen_link', 'No article found')}

Context: {rag_context.get('context', '')}
Available Steps: {', '.join(rag_context.get('raw_steps', []))}
"""

        payload = {
            "messages": [
                {"role": "system", "content": "You are a Tally support assistant. Provide structured responses based on knowledge base context."},
                {"role": "user", "content": prompt}
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 400,
                "temperature": 0.1,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "model_id": model_id,
            "project_id": project_id
        }

        result = make_watsonx_request(payload, api_key)
        if result:
            raw_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Clean and format response
            formatted_response = clean_and_format_response(raw_response, filename)
            
            return ResolutionResponse(
                resolution=formatted_response,
                success=True,
                message="RAG-enhanced resolution generated successfully"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolution generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resolution generation failed: {str(e)}")

# Add helper function for cleaning output
def clean_and_format_response(text: str, filename: str) -> str:
    """Clean and format the response"""
    # Remove duplicates
    if "END OF ANSWER" in text:
        text = text.split("END OF ANSWER")[0].strip()
    
    # Format with markdown
    text = re.sub(r'(?i)^File:\s*', f'**File:** {filename}\n\n**Identified Intent:** ', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^Identified\s+Intent:\s*', '**Identified Intent:** ', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^Recommended\s+Resolution\s+Path:\s*', '**Recommended Resolution Path:**\n\n', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^Suggested\s+Knowledge\s+Base\s+Article:\s*', '**Suggested Knowledge Base Article:**\n\n', text, flags=re.MULTILINE)
    
    return text.strip()

@app.get("/test-rag-status")
async def test_rag_status():
    """Test RAG pipeline status"""
    try:
        test_results = {
            "embeddings_loaded": bool(knowledge_retriever.sbert),
            "chroma_connected": bool(knowledge_retriever.chroma_client),
            "collections_available": {
                "links": bool(knowledge_retriever.links_collection),
                "content": bool(knowledge_retriever.content_collection)
            }
        }
        
        if knowledge_retriever.links_collection:
            test_results["collections_available"]["links_count"] = knowledge_retriever.links_collection.count()
        if knowledge_retriever.content_collection:
            test_results["collections_available"]["content_count"] = knowledge_retriever.content_collection.count()
            
        return {
            "status": "healthy" if all([
                test_results["embeddings_loaded"],
                test_results["chroma_connected"]
            ]) else "degraded",
            "components": test_results,
            "message": "RAG pipeline status check completed"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "RAG pipeline test failed"
        }
@app.post("/analyze", response_model=FullAnalysisResponse)
async def full_analysis_endpoint(file: UploadFile = File(...)):
    """Complete analysis with RAG enhancement: transcription, intent recognition, and resolution generation."""
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")

        # Step 1: Transcribe audio using current STT model
        try:
            current_model = get_current_stt_model()
            logger.info(f"Using STT model for full analysis: {current_model}")
            
            if current_model == "whisper":
                transcription = transcribe_audio_file_whisper(file)
            elif current_model == "assisto":
                transcription = transcribe_audio_file_assisto(file)
            else:  # Default to kibo
                transcription = transcribe_audio_file(file)
        except APIError as api_err:
            logger.error(f"Transcription API error: {str(api_err)}")
            if api_err.status_code == 504:
                raise HTTPException(status_code=504, detail="Audio transcription timed out.")
            else:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(api_err)}")

        # Step 2: Recognize intent (your existing intent recognition)
        try:
            intent_result = await recognize_intent_endpoint({"text": transcription})
        except HTTPException as intent_err:
            logger.error(f"Intent recognition error: {str(intent_err)}")
            intent_result = IntentResponse(
                intent="Intent recognition failed",
                success=False,
                message=str(intent_err.detail)
            )

        # Step 3: Extract keywords and entities
        try:
            keyword_result = await extract_keywords_endpoint({"text": transcription})
        except HTTPException as keyword_err:
            logger.error(f"Keyword extraction error: {str(keyword_err)}")
            keyword_result = KeywordExtractionResponse(
                keywords=[],
                overall_confidence=0,
                success=False,
                message=str(keyword_err.detail)
            )

        # Step 4: Generate RAG-enhanced resolution
        try:
            resolution_result = await generate_resolution_endpoint({
                "content": transcription,
                "intent": intent_result.intent,
                "filename": file.filename  # Pass filename for proper formatting
            })
        except HTTPException as resolution_err:
            logger.error(f"Resolution generation error: {str(resolution_err)}")
            resolution_result = ResolutionResponse(
                resolution="Resolution generation failed",
                success=False,
                message=str(resolution_err.detail)
            )

        # Prepare keywords data for response
        keywords_data = {
            "keywords": keyword_result.keywords,
            "overall_confidence": keyword_result.overall_confidence
        }

        return FullAnalysisResponse(
            transcription=transcription,
            intent=intent_result.intent,
            resolution=resolution_result.resolution,
            keywords=keywords_data,
            success=True,
            message="Complete analysis with keyword extraction completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Full analysis failed: {str(e)}")

# Utility function for testing service connectivity with retry
@retry_with_backoff(max_retries=2, base_delay=1, max_delay=10)
def test_service_connectivity(url: str, headers: dict = None, timeout: int = 10) -> dict:
    """Test service connectivity with retry logic."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        return {
            "connected": True,
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
            "headers": dict(response.headers)
        }
    except requests.exceptions.Timeout:
        raise APIError("Service connection timed out", status_code=504)
    except requests.exceptions.ConnectionError:
        raise APIError("Service connection failed", status_code=503)
    except Exception as e:
        raise APIError(f"Service connection error: {str(e)}", status_code=500)

# Enhanced error handling for file operations
def safe_file_operation(operation_name: str, operation_func, *args, **kwargs):
    """Safely execute file operations with error handling."""
    try:
        return operation_func(*args, **kwargs)
    except PermissionError:
        raise APIError(f"Permission denied for {operation_name}", status_code=403)
    except FileNotFoundError:
        raise APIError(f"File not found for {operation_name}", status_code=404)
    except OSError as e:
        raise APIError(f"File system error during {operation_name}: {str(e)}", status_code=500)
    except Exception as e:
        raise APIError(f"Unexpected error during {operation_name}: {str(e)}", status_code=500)

# Enhanced error handling for audio file processing
def validate_audio_file(file: UploadFile) -> tuple[bytes, float]:
    """Validate and read audio file with comprehensive error checking."""
    try:
        # Check file extension
        if not file.filename.lower().endswith('.mp3'):
            raise HTTPException(status_code=400, detail="Only MP3 files are supported")
        
        # Read file content
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > MAX_FILE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.2f} MB (max {MAX_FILE_MB} MB)"
            )
        
        return content, file_size_mb
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")

# Enhanced logging for debugging
def log_api_request(api_name: str, request_type: str, details: dict = None):
    """Log API request details for debugging."""
    log_data = {
        "timestamp": time.time(),
        "api": api_name,
        "request_type": request_type,
        "details": details or {}
    }
    logger.info(f"API Request: {log_data}")

def log_api_response(api_name: str, response_type: str, status_code: int, details: dict = None):
    """Log API response details for debugging."""
    log_data = {
        "timestamp": time.time(),
        "api": api_name,
        "response_type": response_type,
        "status_code": status_code,
        "details": details or {}
    }
    logger.info(f"API Response: {log_data}")

# Enhanced error messages for different scenarios
ERROR_MESSAGES = {
    "timeout": {
        "kibo": "Kibo AI transcription service is taking longer than expected. This may happen with large audio files.",
        "watson": "WatsonX analysis is taking longer than expected. The request may be too complex.",
        "general": "The request is taking longer than expected. Please try again with a smaller file or simpler request."
    },
    "connection": {
        "kibo": "Unable to connect to Kibo AI transcription service. Please check your internet connection.",
        "watson": "Unable to connect to WatsonX service. Please check your internet connection.",
        "chroma": "Unable to connect to ChromaDB database. Please check your database configuration.",
        "general": "Connection failed. Please check your internet connection and try again."
    },
    "rate_limit": {
        "kibo": "Kibo AI service is experiencing high load. Please wait before making another request.",
        "watson": "WatsonX service is experiencing high load. Please wait before making another request.",
        "general": "Service is experiencing high load. Please wait before making another request."
    },
    "service_unavailable": {
        "kibo": "Kibo AI transcription service is temporarily unavailable. Please try again later.",
        "watson": "WatsonX service is temporarily unavailable. Please try again later.",
        "chroma": "ChromaDB database is temporarily unavailable. Please try again later.",
        "general": "Service is temporarily unavailable. Please try again later."
    }
}

def get_error_message(error_type: str, service: str = "general") -> str:
    """Get appropriate error message based on error type and service."""
    return ERROR_MESSAGES.get(error_type, {}).get(service, ERROR_MESSAGES.get(error_type, {}).get("general", "An error occurred. Please try again."))

# Enhanced retry configuration for different services
RETRY_CONFIGS = {
    "kibo": {"max_retries": 3, "base_delay": 3, "max_delay": 60},
    "watson": {"max_retries": 3, "base_delay": 2, "max_delay": 30},
    "chroma": {"max_retries": 3, "base_delay": 2, "max_delay": 30},
    "general": {"max_retries": 2, "base_delay": 1, "max_delay": 10}
}

def get_retry_config(service: str) -> dict:
    """Get retry configuration for specific service."""
    return RETRY_CONFIGS.get(service, RETRY_CONFIGS["general"])

# Enhanced timeout configuration for different operations
TIMEOUT_CONFIGS = {
    "kibo_transcription": {
        "small_file": 300,    # 5 minutes for files < 10MB
        "medium_file": 600,   # 10 minutes for files 10-25MB
        "large_file": 900     # 15 minutes for files > 25MB
    },
    "watson_analysis": {
        "intent": 120,        # 2 minutes for intent recognition
        "resolution": 300,    # 5 minutes for resolution generation
        "full_analysis": 600  # 10 minutes for full analysis
    },
    "chroma_operation": {
        "search": 30,         # 30 seconds for search
        "upload": 120,        # 2 minutes for upload
        "retrieve": 30        # 30 seconds for retrieve
    }
}

def get_timeout_config(operation: str, file_size_mb: float = None) -> int:
    """Get appropriate timeout for specific operation."""
    if operation == "kibo_transcription" and file_size_mb:
        if file_size_mb < 10:
            return TIMEOUT_CONFIGS["kibo_transcription"]["small_file"]
        elif file_size_mb < 25:
            return TIMEOUT_CONFIGS["kibo_transcription"]["medium_file"]
        else:
            return TIMEOUT_CONFIGS["kibo_transcription"]["large_file"]
    
    # Get timeout for specific operation
    for category, configs in TIMEOUT_CONFIGS.items():
        if operation in configs:
            return configs[operation]
    
    # Default timeout
    return 300

# Enhanced monitoring and alerting
class ServiceMonitor:
    """Monitor service health and performance."""
    
    def __init__(self):
        self.service_stats = {}
        self.error_counts = {}
        self.last_check = {}
    
    def record_request(self, service: str, success: bool, response_time: float = None):
        """Record service request statistics."""
        if service not in self.service_stats:
            self.service_stats[service] = {"total": 0, "success": 0, "failed": 0, "avg_response_time": 0}
        
        stats = self.service_stats[service]
        stats["total"] += 1
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            # Track error count
            if service not in self.error_counts:
                self.error_counts[service] = 0
            self.error_counts[service] += 1
        
        if response_time:
            # Update average response time
            current_avg = stats["avg_response_time"]
            total_requests = stats["total"]
            stats["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
    
    def get_service_health(self, service: str) -> dict:
        """Get health metrics for a specific service."""
        if service not in self.service_stats:
            return {"status": "unknown", "message": "No data available"}
        
        stats = self.service_stats[service]
        total = stats["total"]
        success_rate = (stats["success"] / total * 100) if total > 0 else 0
        
        if success_rate >= 95:
            status = "healthy"
        elif success_rate >= 80:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "total_requests": total,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(stats["avg_response_time"], 3),
            "error_count": self.error_counts.get(service, 0)
        }
    
    def should_retry(self, service: str) -> bool:
        """Determine if service should be retried based on recent performance."""
        health = self.get_service_health(service)
        return health["status"] != "unhealthy" and self.error_counts.get(service, 0) < 5

# Initialize service monitor
service_monitor = ServiceMonitor()

# Enhanced error handling with service monitoring
def handle_service_error(service: str, error: Exception, operation: str = "unknown"):
    """Handle service errors with monitoring and logging."""
    error_type = type(error).__name__
    error_message = str(error)
    
    # Record the error
    service_monitor.record_request(service, success=False)
    
    # Log detailed error information
    logger.error(f"Service error in {service} during {operation}: {error_type}: {error_message}")
    
    # Determine if retry should be attempted
    if service_monitor.should_retry(service):
        logger.info(f"Service {service} is healthy enough to retry operation {operation}")
        return True
    else:
        logger.warning(f"Service {service} has too many errors, not retrying operation {operation}")
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






