import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- ABSOLUTE TOP OF YOUR MAIN SCRIPT ---
if getattr(sys, 'frozen', False):
    bundle_root = sys._MEIPASS
else:
    # Fallback for non-bundled execution (e.g., during development)
    bundle_root = os.path.dirname(os.path.abspath(__file__))

os.environ["PHONEMIZER_ESPEAK_DATA_PATH"] = os.path.join(bundle_root, 'espeakng_loader/espeak-ng-data')
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(bundle_root, 'espeakng_loader/libespeak-ng.dylib')

import io
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import unquote
import asyncio # For async background task
import time    # For timestamp tracking
from typing import Union, Tuple


# Assuming kokoro_onnx.py is in the same directory or accessible via your Python path
try:
    from kokoro_onnx import Kokoro
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from kokoro_onnx import Kokoro


# --- Configuration ---
KOKORO_SERVER_PORT = int(os.environ.get("KOKORO_SERVER_PORT", "3456"))
KOKORO_LOG_FILE = os.environ.get("KOKORO_LOG_FILE", "kokoro_server.log")
KOKORO_LOG_LEVEL = os.environ.get("KOKORO_LOG_LEVEL", "INFO").upper()

KOKORO_PING_CHECK_INTERVAL_SECONDS = float(os.environ.get("KOKORO_PING_CHECK_INTERVAL_SECONDS", "5.0"))
KOKORO_PING_TIMEOUT_SECONDS = float(os.environ.get("KOKORO_PING_TIMEOUT_SECONDS", "20.0"))

KOKORO_MODEL_POOL_SIZE = int(os.environ.get("KOKORO_MODEL_POOL_SIZE", "2"))
KOKORO_MODEL_POOL_TIMEOUT_MINUTES = float(os.environ.get("KOKORO_MODEL_POOL_TIMEOUT_MINUTES", "5.0"))

KOKORO_ENABLE_HEARTBEAT = os.environ.get("KOKORO_ENABLE_HEARTBEAT", "True").lower() == "true"


# --- Logging Configuration ---
LOG_MAX_BYTES = 5 * 1024 * 1024 # 5 MB
LOG_BACKUP_COUNT = 3 # Keep 3 backup log files

logger = logging.getLogger(__name__)
logger.setLevel(KOKORO_LOG_LEVEL)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(KOKORO_LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# --- Global Model Pool ---
# Stores {model_id: {"instance": Kokoro_instance, "last_used": float, "lock": asyncio.Lock()}}
MODEL_POOL: dict[str, dict[str, Union[Kokoro, float, asyncio.Lock]]] = {}


# --- Heartbeat Global ---
LAST_PING_TIMESTAMP: float | None = None


# --- Heartbeat Monitor Task ---
async def heartbeat_monitor_task():
    global LAST_PING_TIMESTAMP
    logger.info("Heartbeat monitor started.")

    while True:
        await asyncio.sleep(KOKORO_PING_CHECK_INTERVAL_SECONDS)

        if LAST_PING_TIMESTAMP is None:
            logger.debug("No ping received yet. Waiting for initial ping.")
            continue

        time_since_last_ping = time.time() - LAST_PING_TIMESTAMP
        if time_since_last_ping > KOKORO_PING_TIMEOUT_SECONDS:
            logger.critical(f"No heartbeat ping received for {time_since_last_ping:.2f} seconds (timeout: {KOKORO_PING_TIMEOUT_SECONDS}s). Assuming main app is gone. Shutting down server.")
            sys.exit(0)
        else:
            logger.debug(f"Last ping received {time_since_last_ping:.2f} seconds ago. All clear.")


# --- Model Pool Eviction Task ---
async def model_pool_eviction_task():
    """
    Periodically checks the model pool and unloads models that haven't been used
    within KOKORO_MODEL_POOL_TIMEOUT_MINUTES and are not currently busy.
    """
    global MODEL_POOL
    timeout_seconds = KOKORO_MODEL_POOL_TIMEOUT_MINUTES * 60
    logger.info(f"Model pool eviction monitor started with timeout: {timeout_seconds} seconds.")

    while True:
        await asyncio.sleep(60) # Check every minute

        current_time = time.time()
        models_to_evict = []

        # It's safer to iterate over a copy of keys if you're deleting elements
        for model_id in list(MODEL_POOL.keys()):
            model_entry = MODEL_POOL.get(model_id)
            if model_entry is None: # Model might have been removed by another process/request
                continue

            # Attempt to acquire lock without blocking. If successful, model is not busy.
            # If not successful, model is busy, skip it.
            if model_entry["lock"].locked():
                logger.debug(f"Model '{model_id}' is currently busy, skipping eviction check.")
                continue

            last_used_time = model_entry["last_used"]
            if (current_time - last_used_time) > timeout_seconds:
                models_to_evict.append(model_id)

        for model_id in models_to_evict:
            # Re-check if model is busy just before attempting to delete it
            model_entry = MODEL_POOL.get(model_id)
            if model_entry and not model_entry["lock"].locked():
                logger.info(f"Evicting model '{model_id}' from pool due to timeout.")
                del MODEL_POOL[model_id] # Allow garbage collection
            elif model_entry and model_entry["lock"].locked():
                logger.debug(f"Model '{model_id}' became busy just before eviction, skipping this cycle.")
        
        if models_to_evict: # Log only if any models were considered for eviction
            logger.info(f"Model pool current size: {len(MODEL_POOL)}")


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global LAST_PING_TIMESTAMP

    logger.info(f"Application startup: Listening on port: {KOKORO_SERVER_PORT}")
    
    # --- Startup Logic ---
    try:
        if KOKORO_ENABLE_HEARTBEAT:
            LAST_PING_TIMESTAMP = time.time() 
            logger.info("Heartbeat timer initialized.")
            asyncio.create_task(heartbeat_monitor_task())
            logger.info("Heartbeat monitor task scheduled.")
        else:
            logger.info("Heartbeat monitor disabled as per KOKORO_ENABLE_HEARTBEAT setting.")

        asyncio.create_task(model_pool_eviction_task())
        logger.info("Model pool eviction task scheduled.")
        
        yield

    except Exception as e:
        logger.critical(f"Failed during server startup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize during startup: {e}") 

    finally:
        logger.info("Application shutdown: Cleaning up resources...")
        MODEL_POOL.clear()
        logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Kokoro TTS API",
    description="A simple API for Kokoro Text-to-Speech synthesis with model pooling",
    version="1.4.0", # Version updated for new endpoints
    lifespan=lifespan
)

# --- Request Body Models ---
class TTSRequest(BaseModel):
    text: str
    voice: str # voice ID (e.g., "af_heart", "zm_097") - still needed for Kokoro.create()
    speed: float = 1.0
    lang: str = "en-us" # This might be derived from voice, but kept for compatibility
    
    # These are now provided directly by the macOS app
    model_id: str # Unique ID for the model (e.g., "kokoro-v1.2-zh") - for pooling
    model_onnx_path: str # Absolute path to the .onnx model file
    model_voice_bin_path: str # Absolute path to the .bin voice data file

class LoadModelRequest(BaseModel):
    model_id: str
    model_onnx_path: str
    model_voice_bin_path: str

class UnloadModelRequest(BaseModel):
    model_id: str


# --- Shared Model Loading Logic ---
async def _ensure_model_in_pool(
    model_id: str,
    model_onnx_path: str,
    model_voice_bin_path: str,
    purpose: str = "synthesis" # Added for better logging
) -> Tuple[Kokoro, asyncio.Lock, bool]:
    """
    Ensures the specified model is loaded into the MODEL_POOL.
    Returns the Kokoro instance, its lock, and a boolean indicating if it was newly loaded.
    Raises HTTPException on failure.
    """
    is_newly_loaded = False
    
    # Validate provided paths
    if not os.path.isfile(model_onnx_path):
        logger.critical(f"Error ({purpose}): Model ONNX file not found at path: {model_onnx_path}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model ONNX file not found at provided path: {model_onnx_path}")
    if not os.path.isfile(model_voice_bin_path):
        logger.critical(f"Error ({purpose}): Voice BIN file not found at path: {model_voice_bin_path}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Voice BIN file not found at provided path: {model_voice_bin_path}")

    if model_id in MODEL_POOL:
        model_entry = MODEL_POOL[model_id]
        kokoro_instance = model_entry["instance"]
        model_lock = model_entry["lock"]
        # Only update timestamp here. The actual "last used for synthesis" update is within the lock for synthesize endpoint.
        # For load_model, this is sufficient.
        model_entry["last_used"] = time.time() 
        logger.debug(f"Using existing model '{model_id}' from pool for {purpose}.")
    else:
        logger.info(f"Model '{model_id}' not in pool. Attempting to load from provided paths for {purpose}.")
        is_newly_loaded = True
        if len(MODEL_POOL) >= KOKORO_MODEL_POOL_SIZE:
            # Find LRU model that is not busy for eviction
            lru_model_id = None
            lru_time = float('inf')
            
            # Iterate over a copy of keys to safely find LRU while other operations might modify MODEL_POOL
            for mid in list(MODEL_POOL.keys()):
                entry = MODEL_POOL.get(mid)
                if entry and not entry["lock"].locked(): # Only consider non-busy models for eviction
                    if entry["last_used"] < lru_time:
                        lru_time = entry["last_used"]
                        lru_model_id = mid
            
            if lru_model_id:
                logger.info(f"Pool full ({len(MODEL_POOL)} models). Evicting LRU (and not busy) model: '{lru_model_id}' to load '{model_id}'.")
                del MODEL_POOL[lru_model_id] # Allow garbage collection
            else:
                # This case means pool is full and all models are currently busy.
                logger.warning(f"Model pool is full ({len(MODEL_POOL)} models) and all models are busy. Cannot load new model '{model_id}'.")
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Model pool exhausted: all models are currently in use. Please try again shortly.")

        try:
            kokoro_instance = Kokoro(model_onnx_path, model_voice_bin_path)
            model_lock = asyncio.Lock()
            MODEL_POOL[model_id] = {
                "instance": kokoro_instance,
                "last_used": time.time(),
                "lock": model_lock
            }
            logger.info(f"Model '{model_id}' loaded and added to pool. Current pool size: {len(MODEL_POOL)}")

        except (FileNotFoundError, RuntimeError) as e:
            logger.critical(f"Failed to load model '{model_id}' from provided paths: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load TTS model: {e}. Check provided paths and file integrity.")
        except Exception as e:
            logger.critical(f"Unexpected error loading model '{model_id}' from provided paths: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred while loading model: {e}")
    
    return kokoro_instance, model_lock, is_newly_loaded


# --- API Endpoints ---
@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesizes speech from text using the Kokoro TTS engine.
    The macOS app provides the model_id and direct file paths.
    Returns the audio as a WAV file. Models are loaded on demand and managed in a pool.
    """
    if not request.text.strip():
        logger.warning("Synthesis requested with empty text.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text cannot be empty.")

    # Ensure the model is loaded/available in the pool
    kokoro_instance, model_lock, _ = await _ensure_model_in_pool(
        request.model_id, request.model_onnx_path, request.model_voice_bin_path, purpose="synthesis"
    )

    # Acquire the lock for the model while it's being used for synthesis
    async with model_lock:
        # Update last used timestamp immediately upon acquiring the lock
        # This ensures that even long syntheses keep the model "fresh" in the pool
        MODEL_POOL[request.model_id]["last_used"] = time.time()
        
        try:
            samples, sample_rate = kokoro_instance.create(
                request.text,
                voice=request.voice,
                speed=request.speed,
                lang=request.lang
            )

            buffer = io.BytesIO()
            sf.write(buffer, samples, sample_rate, format='WAV')
            buffer.seek(0)
            logger.info(f"Successfully synthesized text for voice '{request.voice}' using model '{request.model_id}'.")
            return Response(content=buffer.getvalue(), media_type="audio/wav")

        except Exception as e:
            logger.error(f"Synthesis failed for text: '{request.text[:50]}...' with error: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to synthesize speech: {e}")

@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    """
    Explicitly loads a model into the pool. Useful for warming up or pre-loading.
    """
    _, _, is_newly_loaded = await _ensure_model_in_pool(
        request.model_id, request.model_onnx_path, request.model_voice_bin_path, purpose="explicit load"
    )

    if is_newly_loaded:
        return {"status": "success", "message": f"Model '{request.model_id}' loaded successfully."}
    else:
        return {"status": "success", "message": f"Model '{request.model_id}' already loaded and timestamp updated."}

@app.post("/unload_model")
async def unload_model(request: UnloadModelRequest):
    """
    Explicitly unloads a model from the pool.
    Will not unload if the model is currently busy.
    """
    if request.model_id not in MODEL_POOL:
        logger.warning(f"Attempted to unload non-existent model: '{request.model_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{request.model_id}' not found in pool.")
    
    model_entry = MODEL_POOL[request.model_id]
    if model_entry["lock"].locked():
        logger.warning(f"Cannot unload model '{request.model_id}': It is currently busy.")
        return {"status": "failed", "message": f"Model '{request.model_id}' is currently in use and cannot be unloaded."}
    
    del MODEL_POOL[request.model_id]
    logger.info(f"Model '{request.model_id}' explicitly unloaded from pool. Current pool size: {len(MODEL_POOL)}")
    return {"status": "success", "message": f"Model '{request.model_id}' unloaded successfully."}


@app.get("/health")
async def health_check():
    """
    Checks the health of the TTS service and reports on the model pool.
    """
    # Count how many models are busy
    busy_models = sum(1 for entry in MODEL_POOL.values() if entry["lock"].locked())

    pool_status = {
        "status": "ok",
        "message": "Kokoro TTS service is running.",
        "active_models_in_pool": len(MODEL_POOL),
        "busy_models_in_pool": busy_models,
        "max_pool_size": KOKORO_MODEL_POOL_SIZE,
        "model_timeout_minutes": KOKORO_MODEL_POOL_TIMEOUT_MINUTES,
        "loaded_models": list(MODEL_POOL.keys())
    }
    logger.debug(f"Health check successful: {pool_status}")
    return pool_status

@app.post("/ping")
async def receive_ping():
    """
    Receives a heartbeat ping from the main application and resets the internal timer.
    """
    global LAST_PING_TIMESTAMP
    LAST_PING_TIMESTAMP = time.time()
    logger.debug("Received heartbeat ping. Timer reset.")
    return {"status": "ok", "message": "Pong!"}


if __name__ == "__main__":
    logger.info(f"Starting kokoro_server. PID: {os.getpid()}")

    if KOKORO_SERVER_PORT == 0:
        logger.critical("Error: KOKORO_SERVER_PORT environment variable not set or invalid. Cannot start server.")
        sys.exit(1)
    
    logger.info(f"Starting server on TCP/IP Port: {KOKORO_SERVER_PORT}")

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.addHandler(stream_handler)
    uvicorn_logger.setLevel(KOKORO_LOG_LEVEL)

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers = []
    uvicorn_access_logger.addHandler(file_handler)
    uvicorn_access_logger.addHandler(stream_handler)
    uvicorn_access_logger.setLevel(KOKORO_LOG_LEVEL)

    try:
        uvicorn.run(app, host="127.0.0.1", port=KOKORO_SERVER_PORT, lifespan="on", log_level=KOKORO_LOG_LEVEL.lower())
        logger.info("Uvicorn server stopped gracefully.")
    except Exception as e:
        logger.critical(f"Error during Uvicorn server run: {e}", exc_info=True)
        sys.exit(1)