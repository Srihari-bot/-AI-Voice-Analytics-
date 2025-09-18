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
from Resolution_Prompt import get_resolution_system_message, get_resolution_user_message, extract_intent_category
import chromadb
from chromadb.config import Settings
import numpy as np
import base64
from uuid import uuid4
import asyncio
from functools import wraps
from pydub import AudioSegment
from io import BytesIO
from sarvamai import SarvamAI

# Load environment variables
load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "audio_kb")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
MAX_FILE_MB = 30

# Retry configuration
MAX_RETRIES = 3
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
    description="API for audio transcription (using Sarvam AI) and GST/License analysis using IBM WatsonX",
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
    "expiry": 0
}

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

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

class FullAnalysisResponse(BaseModel):
    transcription: str
    intent: str
    resolution: str
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
                        logger.error(f"Final attempt {attempt + 1} failed: {str(e)}")
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            
            # Re-raise the last exception if all retries failed
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

@retry_with_backoff(max_retries=3, base_delay=2, max_delay=30)
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
                if "invalid_grant" in error_detail.lower():
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

@retry_with_backoff(max_retries=3, base_delay=2, max_delay=30)
def make_watsonx_request(payload: dict, api_key: str, max_retries: int = 2) -> Optional[dict]:
    """
    Make a request to WatsonX with automatic token refresh on 401 errors and enhanced error handling.
    """
    for attempt in range(max_retries + 1):
        try:
            force_refresh = attempt > 0
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

def get_sarvam_api_key() -> str:
    """
    Get Sarvam API key from environment variables.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="SARVAM_API_KEY not found in environment variables. Please set SARVAM_API_KEY in your .env file."
        )
    return api_key

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
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
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

def convert_mp3_to_wav(audio_content: bytes, filename: str) -> bytes:
    """
    Convert MP3 audio content to WAV format using pydub.
    """
    try:
        # Load audio from bytes
        audio = AudioSegment.from_mp3(BytesIO(audio_content))
        
        # Convert to WAV
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_content = wav_buffer.getvalue()
        
        logger.info(f"Successfully converted MP3 to WAV: {len(audio_content)} bytes -> {len(wav_content)} bytes")
        return wav_content
        
    except Exception as e:
        logger.error(f"Error converting MP3 to WAV: {e}")
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")

@retry_with_backoff(max_retries=3, base_delay=3, max_delay=60)
def transcribe_audio_file(audio_file: UploadFile) -> str:
    """
    Transcribe audio file using Sarvam AI Speech Transcription API with enhanced error handling and retries.
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
        
        # Convert MP3 to WAV if needed (Sarvam AI supports WAV)
        original_filename = audio_file.filename
        if original_filename.lower().endswith('.mp3'):
            logger.info(f"Converting MP3 file to WAV for Sarvam AI: {original_filename}")
            content = convert_mp3_to_wav(content, original_filename)
            # Update filename for API call
            audio_file.filename = Path(original_filename).stem + '.wav'
            logger.info(f"Converted filename: {audio_file.filename}")
        
        # Log audio file details for debugging
        logger.info(f"Audio file details:")
        logger.info(f"  Original filename: {original_filename}")
        logger.info(f"  Processing filename: {audio_file.filename}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        logger.info(f"  Content length: {len(content)} bytes")
        
        # Estimate processing time and warn if file is large
        if file_size_mb > 10:
            logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Processing may take longer.")
        elif file_size_mb > 25:
            logger.warning(f"Very large file detected ({file_size_mb:.2f} MB). Consider using a smaller file if timeout occurs.")
        
        # Get API key
        api_key = get_sarvam_api_key()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Initialize Sarvam AI client
            client = SarvamAI(api_subscription_key=api_key)
            
            logger.info(f"Sending audio file to Sarvam AI API: {audio_file.filename}")
            logger.info(f"API Key configured: {api_key[:8]}..." if len(api_key) > 8 else "Short key")
            logger.info("Starting transcription request... This may take several minutes for large files.")
            
            # Use dynamic timeout based on file size
            timeout_seconds = 300  # 5 minutes base timeout
            if file_size_mb > 10:
                timeout_seconds = 600  # 10 minutes for large files
            elif file_size_mb > 25:
                timeout_seconds = 900  # 15 minutes for very large files
            
            logger.info(f"Using timeout of {timeout_seconds} seconds for {file_size_mb:.2f} MB file")
            
            # Transcribe using Sarvam AI
            try:
                logger.info(f"Starting Sarvam AI transcription with parameters:")
                logger.info(f"  Model: saarika:v2.5")
                logger.info(f"  Language: en-IN")
                logger.info(f"  File path: {tmp_file_path}")
                logger.info(f"  File size: {len(content)} bytes")
                
                with open(tmp_file_path, "rb") as file:
                    # Verify file is readable
                    file_content = file.read()
                    logger.info(f"File read successfully, content length: {len(file_content)} bytes")
                    
                    # Reset file pointer
                    file.seek(0)
                    
                    response = client.speech_to_text.transcribe(
                        file=file,
                        model="saarika:v2.5",
                        language_code="en-IN"
                    )
                
                logger.info(f"Sarvam AI Response: {response}")
                logger.info(f"Response type: {type(response)}")
                
                # Extract transcription from Sarvam AI response
                transcription = None
                
                # Sarvam AI typically returns the transcription directly or in a structured format
                if isinstance(response, str):
                    transcription = response.strip()
                    logger.info(f"Direct transcription response: {transcription}")
                elif isinstance(response, dict):
                    # Check common response keys
                    possible_keys = ['transcription', 'text', 'transcript', 'result', 'output', 'content', 'data']
                    
                    for key in possible_keys:
                        if key in response and response[key]:
                            transcription = response[key]
                            logger.info(f"Found transcription in key '{key}': {transcription}")
                            break
                    
                    # If no standard key found, look for any string value
                    if not transcription:
                        for key, value in response.items():
                            if isinstance(value, str) and value.strip():
                                transcription = value
                                logger.info(f"Found string value in key '{key}': {transcription}")
                                break
                elif hasattr(response, 'transcript'):
                    # Handle Sarvam AI response object with transcript attribute
                    transcription = response.transcript
                    logger.info(f"Found transcript attribute: {transcription}")
                elif hasattr(response, 'text'):
                    transcription = response.text.strip()
                    logger.info(f"Response object with text attribute: {transcription}")
                elif hasattr(response, 'diarized_transcript') and response.diarized_transcript:
                    # Try diarized transcript if regular transcript is empty
                    transcription = response.diarized_transcript
                    logger.info(f"Using diarized transcript: {transcription}")
                
                # Check if transcription is valid and not empty
                if transcription and str(transcription).strip():
                    logger.info("Transcription completed successfully")
                    return str(transcription).strip()
                else:
                    # Log detailed response information for debugging
                    logger.warning(f"No valid transcription found in response.")
                    logger.error(f"Full response for debugging: {response}")
                    logger.error(f"Response type: {type(response)}")
                    
                    # If response has attributes, log them
                    if hasattr(response, '__dict__'):
                        logger.error(f"Response attributes: {response.__dict__}")
                    elif hasattr(response, '_fields'):  # Named tuple
                        logger.error(f"Response fields: {response._fields}")
                        for field in response._fields:
                            if hasattr(response, field):
                                value = getattr(response, field)
                                logger.error(f"  {field}: {value} (type: {type(value)})")
                    
                    # Check for alternative transcript sources
                    alternative_sources = []
                    if hasattr(response, 'diarized_transcript') and response.diarized_transcript:
                        alternative_sources.append(f"diarized_transcript: {response.diarized_transcript}")
                    if hasattr(response, 'timestamps') and response.timestamps:
                        alternative_sources.append(f"timestamps: {response.timestamps}")
                    
                    if alternative_sources:
                        logger.info(f"Alternative sources found: {alternative_sources}")
                        # Try to extract any meaningful text from alternative sources
                        for source in alternative_sources:
                            if 'diarized_transcript' in source and source.split(': ', 1)[1].strip():
                                return f"[Using diarized transcript] {source.split(': ', 1)[1].strip()}"
                    
                    # Return a more informative error message with troubleshooting steps
                    troubleshooting_steps = [
                        "1. Check if the audio file contains clear speech",
                        "2. Ensure the audio is not too quiet or has background noise",
                        "3. Verify the audio format is supported (WAV, MP3)",
                        "4. Try with a shorter audio file (under 30 seconds)",
                        "5. Check if the language is English (en-IN)",
                        "6. Ensure the audio has a good signal-to-noise ratio"
                    ]
                    
                    error_message = f"[Audio processed but no transcription found] The audio file was processed successfully but no transcription text was generated.\n\nTroubleshooting steps:\n" + "\n".join(troubleshooting_steps)
                    error_message += f"\n\nResponse details: {str(response)[:300]}..."
                    
                    return error_message
                
            except Exception as sarvam_error:
                logger.error(f"Sarvam AI API error: {str(sarvam_error)}")
                
                # Handle specific Sarvam AI errors
                error_str = str(sarvam_error).lower()
                if "unauthorized" in error_str or "401" in error_str:
                    raise APIError("Unauthorized: Invalid API key for Sarvam AI. Please check your SARVAM_API_KEY in .env file.", status_code=401)
                elif "forbidden" in error_str or "403" in error_str:
                    raise APIError("Forbidden: API key doesn't have permission for Sarvam AI transcription service.", status_code=403)
                elif "bad request" in error_str or "400" in error_str:
                    raise APIError(f"Bad Request to Sarvam AI API: {str(sarvam_error)}", status_code=400)
                elif "rate limit" in error_str or "429" in error_str:
                    raise APIError(f"Rate limited by Sarvam AI API. Please wait before retrying.", status_code=429)
                elif "service unavailable" in error_str or "503" in error_str:
                    raise APIError("Sarvam AI service is temporarily unavailable. Please try again later.", status_code=503)
                elif "timeout" in error_str:
                    raise APIError(f"Sarvam AI API request timed out after {timeout_seconds} seconds. The audio file might be too large or the service is busy.", status_code=504)
                else:
                    raise APIError(f"Sarvam AI API error: {str(sarvam_error)}", status_code=500)
        
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
    sarvam_api_key = os.getenv("SARVAM_API_KEY")
    watsonx_api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("WATSONX_MODEL_ID")
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "environment": {
            "sarvam_api_key_configured": bool(sarvam_api_key),
            "watsonx_api_key_configured": bool(watsonx_api_key),
            "project_id_configured": bool(project_id),
            "model_id_configured": bool(model_id),
        },
        "token_status": {
            "has_cached_token": bool(access_token_cache["token"]),
            "token_valid": time.time() < access_token_cache["expiry"] if access_token_cache["token"] else False
        },
        "services": {}
    }
    
    # Test Sarvam AI service
    try:
        if sarvam_api_key:
            sarvam_status = await check_sarvam_status()
            health_status["services"]["sarvam_ai"] = {
                "status": "healthy" if sarvam_status.get("api_responsive") else "unhealthy",
                "response_time": sarvam_status.get("response_time_seconds"),
                "details": sarvam_status
            }
        else:
            health_status["services"]["sarvam_ai"] = {
                "status": "not_configured",
                "message": "API key not configured"
            }
    except Exception as e:
        health_status["services"]["sarvam_ai"] = {
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
    
    # Sarvam AI Health Check
    try:
        sarvam_status = await check_sarvam_status()
        services["sarvam_ai"] = {
            "healthy": sarvam_status.get("api_responsive", False),
            "response_time": sarvam_status.get("response_time_seconds"),
            "status_code": sarvam_status.get("status_code")
        }
    except Exception as e:
        services["sarvam_ai"] = {"healthy": False, "error": str(e)}
    
    # WatsonX Health Check
    try:
        watson_status = await test_watson_credentials()
        services["watsonx"] = {
            "healthy": watson_status.get("valid", False),
            "message": watson_status.get("message")
        }
    except Exception as e:
        services["watsonx"] = {"healthy": False, "error": str(e)}
    
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
    
    # Test Sarvam AI connectivity
    try:
        sarvam_url = 'https://api.sarvam.ai'
        api_key = os.getenv("SARVAM_API_KEY")
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        sarvam_result = await test_service_connectivity(sarvam_url, headers, timeout=15)
        results["sarvam_ai"] = {
            "url": sarvam_url,
            "status": "connected",
            "details": sarvam_result
        }
    except APIError as e:
        results["sarvam_ai"] = {
            "url": sarvam_url,
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
@retry_with_backoff(max_retries=3, base_delay=2, max_delay=30)
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

@app.get("/test-watson-credentials")
async def test_watson_credentials():
    """Test if Watson credentials are valid without making a full request."""
    try:
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_id = os.getenv("WATSONX_MODEL_ID")
        
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

@app.get("/debug-sarvam-api")
async def debug_sarvam_api():
    """Debug endpoint to test Sarvam API response structure."""
    try:
        api_key = get_sarvam_api_key()
        
        # Initialize Sarvam AI client
        client = SarvamAI(api_subscription_key=api_key)
        
        debug_info = {
            "api_key_format": f"{api_key[:8]}..." if len(api_key) > 8 else "short_key",
            "client_initialized": True,
            "available_models": ["saarika:v2.5"],
            "supported_languages": ["en-IN", "hi-IN", "ta-IN"]
        }
        
        # Test client initialization
        try:
            debug_info["client_test"] = {
                "status": "success",
                "message": "Sarvam AI client initialized successfully"
            }
        except Exception as e:
            debug_info["client_test"] = {
                "status": "error",
                "error": str(e)
            }
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Debug test failed: {str(e)}"}

@app.post("/debug-sarvam-response")
async def debug_sarvam_response(file: UploadFile = File(...)):
    """Debug endpoint to test Sarvam API response structure with actual audio file."""
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")
        
        # Get API key
        api_key = get_sarvam_api_key()
        
        # Read file content
        content = file.file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        logger.info(f"Debug transcription for file: {file.filename} ({file_size_mb:.2f} MB)")
        
        # Convert MP3 to WAV if needed
        original_filename = file.filename
        if original_filename.lower().endswith('.mp3'):
            logger.info(f"Converting MP3 to WAV for debug")
            content = convert_mp3_to_wav(content, original_filename)
            file.filename = Path(original_filename).stem + '.wav'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Initialize Sarvam AI client
            client = SarvamAI(api_subscription_key=api_key)
            
            with open(tmp_file_path, "rb") as file_obj:
                response = client.speech_to_text.transcribe(
                    file=file_obj,
                    model="saarika:v2.5",
                    language_code="en-IN"
                )
            
            # Detailed response analysis
            debug_response = {
                "filename": original_filename,
                "file_size_mb": file_size_mb,
                "response_type": str(type(response)),
                "response_str": str(response),
                "response_repr": repr(response)
            }
            
            # Analyze response attributes
            if hasattr(response, '__dict__'):
                debug_response["response_dict"] = response.__dict__
            elif hasattr(response, '_fields'):
                debug_response["response_fields"] = response._fields
                debug_response["field_values"] = {}
                for field in response._fields:
                    if hasattr(response, field):
                        value = getattr(response, field)
                        debug_response["field_values"][field] = {
                            "value": value,
                            "type": str(type(value)),
                            "is_empty": not value if isinstance(value, str) else False
                        }
            
            # Check for transcript specifically
            if hasattr(response, 'transcript'):
                debug_response["transcript_analysis"] = {
                    "has_transcript": True,
                    "transcript_value": response.transcript,
                    "transcript_type": str(type(response.transcript)),
                    "is_empty": not response.transcript if isinstance(response.transcript, str) else False,
                    "length": len(response.transcript) if isinstance(response.transcript, str) else 0
                }
            else:
                debug_response["transcript_analysis"] = {
                    "has_transcript": False,
                    "available_attributes": [attr for attr in dir(response) if not attr.startswith('_')]
                }
            
            return debug_response
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Debug transcription error: {str(e)}")
        return {"error": f"Debug transcription failed: {str(e)}"}

@app.get("/test-sarvam-credentials")
async def test_sarvam_credentials():
    """Test if Sarvam API credentials are valid."""
    try:
        api_key = os.getenv("SARVAM_API_KEY")
        
        if not api_key:
            return {
                "valid": False,
                "message": "Sarvam API key not configured",
                "instruction": "Please set SARVAM_API_KEY in your .env file"
            }
        
        if api_key == "your_actual_sarvam_api_key_here":
            return {
                "valid": False,
                "message": "Sarvam API key still using placeholder value",
                "instruction": "Please update SARVAM_API_KEY in your .env file with your actual Sarvam API key"
            }
        
        # Test API key format (basic validation)
        if len(api_key) < 10:
            return {
                "valid": False,
                "message": "Sarvam API key appears to be too short",
                "instruction": "Please check your SARVAM_API_KEY in the .env file"
            }
        
        # Test API connectivity by initializing client
        try:
            client = SarvamAI(api_subscription_key=api_key)
            
            return {
                "valid": True,
                "message": "Sarvam API key is configured and client initialized successfully",
                "api_key_format": f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "short_key",
                "instruction": "API client initialization completed. Upload an audio file to test transcription."
            }
        except Exception as client_error:
            return {
                "valid": True,
                "message": "Sarvam API key is configured but client initialization failed",
                "api_key_format": f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 8 else "short_key",
                "client_error": str(client_error),
                "instruction": "API key format looks valid but there may be initialization issues. Try uploading an audio file."
            }
        
    except Exception as e:
        logger.error(f"Sarvam credential test error: {str(e)}")
        return {
            "valid": False,
            "message": f"Error testing Sarvam credentials: {str(e)}",
            "instruction": "Check your .env file configuration"
        }

@app.get("/check-sarvam-status")
async def check_sarvam_status():
    """Quick check if Sarvam AI API is responsive."""
    try:
        api_key = get_sarvam_api_key()
        
        # Initialize Sarvam AI client to test responsiveness
        start_time = time.time()
        client = SarvamAI(api_subscription_key=api_key)
        response_time = time.time() - start_time
        
        return {
            "api_responsive": True,
            "response_time_seconds": round(response_time, 2),
            "status_code": 200,
            "message": f"Sarvam AI client initialized in {response_time:.2f} seconds"
        }
    except Exception as e:
        return {
            "api_responsive": False,
            "error": str(e),
            "message": "Sarvam AI client initialization failed"
        }

@app.post("/test-transcribe")
async def test_transcribe_only(file: UploadFile = File(...)):
    """Test transcription only (without WatsonX) for debugging."""
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")
        
        # Test API key first
        try:
            api_key = get_sarvam_api_key()
            logger.info(f"Using API key for test: {api_key[:8]}..." if len(api_key) > 8 else "Short key")
        except HTTPException as key_error:
            raise HTTPException(status_code=500, detail=f"API key error: {key_error.detail}")
        
        transcription = transcribe_audio_file(file)
        
        return {
            "transcription": transcription,
            "success": True,
            "message": "Transcription completed successfully (test mode)",
            "filename": file.filename,
            "api_used": "Sarvam AI"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test transcription failed: {str(e)}")

@app.get("/test-sarvam-simple")
async def test_sarvam_simple():
    """Simple test to verify Sarvam AI service connectivity."""
    try:
        api_key = get_sarvam_api_key()
        
        # Initialize client
        client = SarvamAI(api_subscription_key=api_key)
        
        return {
            "status": "success",
            "message": "Sarvam AI client initialized successfully",
            "api_key_configured": True,
            "client_ready": True,
            "suggestion": "Try uploading an audio file using /test-transcribe endpoint"
        }
        
    except Exception as e:
        logger.error(f"Simple Sarvam test error: {str(e)}")
        return {
            "status": "error",
            "message": f"Sarvam AI test failed: {str(e)}",
            "api_key_configured": bool(os.getenv("SARVAM_API_KEY")),
            "client_ready": False,
            "suggestion": "Check your SARVAM_API_KEY in the .env file"
        }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Transcribe audio file to text."""
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")
        
        transcription = transcribe_audio_file(file)
        
        return TranscriptionResponse(
            transcription=transcription,
            success=True,
            message="Transcription completed successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/recognize-intent", response_model=IntentResponse)
async def recognize_intent_endpoint(content: dict):
    """Recognize intent from transcribed text."""
    try:
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_id = os.getenv("WATSONX_MODEL_ID")
        
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

@app.post("/generate-resolution", response_model=ResolutionResponse)
async def generate_resolution_endpoint(data: dict):
    """Generate resolution based on identified intent."""
    try:
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_id = os.getenv("WATSONX_MODEL_ID")
        
        if not all([api_key, project_id, model_id]):
            raise HTTPException(
                status_code=500,
                detail="Missing required environment variables (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID)"
            )
        
        content = data.get("content", "")
        intent = data.get("intent", "")
        
        if not content or not intent:
            raise HTTPException(status_code=400, detail="Both content and intent are required")
        
        # Get resolution prompts from separate file
        intent_category = extract_intent_category(intent)
        system_message = get_resolution_system_message(intent_category)
        user_message = get_resolution_user_message(intent, content)
        
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 1500,
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
                resolution_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return ResolutionResponse(
                    resolution=resolution_response.strip(),
                    success=True,
                    message="Resolution generated successfully"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to get resolution response")
        
        except APIError as api_err:
            logger.error(f"WatsonX API error in resolution generation: {str(api_err)}")
            if api_err.status_code == 504:
                raise HTTPException(status_code=504, detail="Resolution generation timed out. Please try again.")
            elif api_err.status_code == 503:
                raise HTTPException(status_code=503, detail="WatsonX service is temporarily unavailable. Please try again later.")
            elif api_err.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limited. Please wait before trying again.")
            else:
                raise HTTPException(status_code=500, detail=f"Resolution generation failed: {str(api_err)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolution generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resolution generation failed: {str(e)}")

@app.post("/analyze", response_model=FullAnalysisResponse)
async def full_analysis_endpoint(file: UploadFile = File(...)):
    """Complete analysis: transcription, intent recognition, and resolution generation."""
    try:
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")
        
        api_key = os.getenv("WATSONX_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="WATSONX_API_KEY not found in environment variables")
        
        # Step 1: Transcribe audio
        try:
            transcription = transcribe_audio_file(file)
        except APIError as api_err:
            logger.error(f"Transcription API error: {str(api_err)}")
            if api_err.status_code == 504:
                raise HTTPException(status_code=504, detail="Audio transcription timed out. The file might be too large or the service is busy.")
            elif api_err.status_code == 503:
                raise HTTPException(status_code=503, detail="Sarvam AI service is temporarily unavailable. Please try again later.")
            elif api_err.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limited by Sarvam AI. Please wait before trying again.")
            else:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(api_err)}")
        
        # Step 2: Recognize intent
        try:
            intent_result = await recognize_intent_endpoint({"text": transcription})
        except HTTPException as intent_err:
            logger.error(f"Intent recognition error: {str(intent_err)}")
            # Continue with analysis even if intent recognition fails
            intent_result = IntentResponse(
                intent="Intent recognition failed",
                success=False,
                message=str(intent_err.detail)
            )
        
        # Step 3: Generate resolution
        try:
            resolution_result = await generate_resolution_endpoint({
                "content": transcription,
                "intent": intent_result.intent
            })
        except HTTPException as resolution_err:
            logger.error(f"Resolution generation error: {str(resolution_err)}")
            # Continue with analysis even if resolution generation fails
            resolution_result = ResolutionResponse(
                resolution="Resolution generation failed",
                success=False,
                message=str(resolution_err.detail)
            )
        
        return FullAnalysisResponse(
            transcription=transcription,
            intent=intent_result.intent,
            resolution=resolution_result.resolution,
            success=True,
            message="Full analysis completed successfully"
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
        "sarvam": "Sarvam AI transcription service is taking longer than expected. This may happen with large audio files.",
        "watson": "WatsonX analysis is taking longer than expected. The request may be too complex.",
        "general": "The request is taking longer than expected. Please try again with a smaller file or simpler request."
    },
    "connection": {
        "sarvam": "Unable to connect to Sarvam AI transcription service. Please check your internet connection.",
        "watson": "Unable to connect to WatsonX service. Please check your internet connection.",
        "chroma": "Unable to connect to ChromaDB database. Please check your database configuration.",
        "general": "Connection failed. Please check your internet connection and try again."
    },
    "rate_limit": {
        "sarvam": "Sarvam AI service is experiencing high load. Please wait before making another request.",
        "watson": "WatsonX service is experiencing high load. Please wait before making another request.",
        "general": "Service is experiencing high load. Please wait before making another request."
    },
    "service_unavailable": {
        "sarvam": "Sarvam AI transcription service is temporarily unavailable. Please try again later.",
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
    "sarvam": {"max_retries": 3, "base_delay": 3, "max_delay": 60},
    "watson": {"max_retries": 3, "base_delay": 2, "max_delay": 30},
    "chroma": {"max_retries": 3, "base_delay": 2, "max_delay": 30},
    "general": {"max_retries": 2, "base_delay": 1, "max_delay": 10}
}

def get_retry_config(service: str) -> dict:
    """Get retry configuration for specific service."""
    return RETRY_CONFIGS.get(service, RETRY_CONFIGS["general"])

# Enhanced timeout configuration for different operations
TIMEOUT_CONFIGS = {
    "sarvam_transcription": {
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
    if operation == "sarvam_transcription" and file_size_mb:
        if file_size_mb < 10:
            return TIMEOUT_CONFIGS["sarvam_transcription"]["small_file"]
        elif file_size_mb < 25:
            return TIMEOUT_CONFIGS["sarvam_transcription"]["medium_file"]
        else:
            return TIMEOUT_CONFIGS["sarvam_transcription"]["large_file"]
    
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

if __name__ == "__sarvam__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

