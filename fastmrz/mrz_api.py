from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmrz import FastMRZ
from fastapi.responses import JSONResponse
import os
import subprocess
import sys
import base64
import logging
from pathlib import Path
from PIL import Image
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def find_tesseract_data_path():
    """
    Dynamically find the correct Tesseract data path and ensure MRZ trained data exists
    """
    possible_paths = [
        # Common Ubuntu/Debian paths
        "/usr/share/tesseract-ocr/5/tessdata/",
        "/usr/share/tesseract-ocr/4.00/tessdata/", 
        "/usr/share/tessdata/",
        "/usr/local/share/tessdata/",
        
        # Common paths on other systems
        "/opt/homebrew/share/tessdata/",  # macOS with Homebrew
        "/usr/local/Cellar/tesseract/*/share/tessdata/",  # macOS Homebrew
        "/Program Files/Tesseract-OCR/tessdata/",  # Windows
        
        # Ubuntu specific paths
        "/usr/share/tesseract-ocr/tessdata/",
        "/snap/tesseract/current/usr/share/tessdata/",  # Snap installation
    ]
    
    # First, try to get tessdata path from tesseract itself
    try:
        result = subprocess.run(['tesseract', '--print-parameters'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'tessdata' in line.lower():
                    # Extract path from tesseract output
                    parts = line.split()
                    for part in parts:
                        if 'tessdata' in part:
                            if os.path.isdir(part):
                                possible_paths.insert(0, part)
        logger.info(f"Tesseract parameters output: {result.stdout}")
    except Exception as e:
        logger.warning(f"Could not get tesseract parameters: {e}")
    
    # Try to get from TESSDATA_PREFIX environment variable
    tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
    if tessdata_prefix:
        possible_paths.insert(0, tessdata_prefix)
        logger.info(f"TESSDATA_PREFIX found: {tessdata_prefix}")
    
    # Check each possible path
    for path in possible_paths:
        if '*' in path:
            # Handle wildcard paths (like Homebrew)
            import glob
            expanded_paths = glob.glob(path)
            for expanded_path in expanded_paths:
                if os.path.isdir(expanded_path):
                    mrz_file = os.path.join(expanded_path, 'mrz.traineddata')
                    if os.path.isfile(mrz_file):
                        logger.info(f"Found tessdata with MRZ at: {expanded_path}")
                        return expanded_path
        else:
            if os.path.isdir(path):
                mrz_file = os.path.join(path, 'mrz.traineddata')
                logger.info(f"Checking path: {path}")
                logger.info(f"MRZ file exists: {os.path.isfile(mrz_file)}")
                if os.path.isfile(mrz_file):
                    return path
    
    logger.warning("No tessdata path with MRZ found")
    return None

def download_mrz_traineddata(tessdata_path):
    """
    Download MRZ trained data if it doesn't exist
    """
    mrz_file = os.path.join(tessdata_path, 'mrz.traineddata')
    if os.path.isfile(mrz_file):
        logger.info(f"MRZ trained data already exists at: {mrz_file}")
        return True
    
    logger.info(f"MRZ trained data not found in {tessdata_path}")
    logger.info("Attempting to download MRZ trained data...")
    
    try:
        import urllib.request
        url = "https://github.com/rahmadmartin/fastmrz/raw/refs/heads/main/tessdata/mrz.traineddata"
        
        # Create directory if it doesn't exist
        os.makedirs(tessdata_path, exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, mrz_file)
        
        if os.path.isfile(mrz_file):
            logger.info(f"✓ MRZ trained data downloaded to {mrz_file}")
            # Check file size
            file_size = os.path.getsize(mrz_file)
            logger.info(f"Downloaded file size: {file_size} bytes")
            return True
        else:
            logger.error(f"✗ Failed to download MRZ trained data to {mrz_file}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error downloading MRZ trained data: {e}")
        return False

def setup_tesseract():
    """
    Setup Tesseract with proper tessdata path
    """
    # Check if tesseract is installed
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        logger.info(f"Tesseract version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
    except Exception as e:
        logger.error(f"Tesseract not found or not working: {e}")
        raise RuntimeError("Tesseract is not installed or not in PATH")
    
    # Find tessdata path
    tessdata_path = find_tesseract_data_path()
    
    if not tessdata_path:
        logger.warning("Could not find tesseract tessdata directory. Trying common locations...")
        # Try to create in a common location
        common_paths = [
            "/usr/share/tesseract-ocr/4.00/tessdata/",
            "/usr/share/tessdata/",
            "/usr/local/share/tessdata/",
            "/tmp/tessdata/"  # Fallback writable location
        ]
        
        for path in common_paths:
            try:
                logger.info(f"Trying to create tessdata at: {path}")
                os.makedirs(path, exist_ok=True)
                if download_mrz_traineddata(path):
                    tessdata_path = path
                    break
            except PermissionError as e:
                logger.warning(f"Permission denied for {path}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error with {path}: {e}")
                continue
    
    if not tessdata_path:
        raise RuntimeError("Could not find or create tesseract tessdata directory")
    
    # Ensure MRZ trained data exists
    if not download_mrz_traineddata(tessdata_path):
        raise RuntimeError(f"Could not download MRZ trained data to {tessdata_path}")
    
    # Set environment variable
    os.environ['TESSDATA_PREFIX'] = tessdata_path
    
    logger.info(f"✓ Tesseract tessdata path set to: {tessdata_path}")
    
    # Verify tesseract can see the MRZ language
    try:
        result = subprocess.run(['tesseract', '--list-langs'], 
                              capture_output=True, text=True, timeout=10)
        available_langs = result.stdout.strip().split('\n')[1:] if result.returncode == 0 else []
        logger.info(f"Available tesseract languages: {available_langs}")
        
        if 'mrz' not in available_langs:
            logger.warning("MRZ language not detected by tesseract!")
    except Exception as e:
        logger.warning(f"Could not check available languages: {e}")
    
    return tessdata_path

def validate_base64_image(base64_string):
    """
    Validate and get info about base64 image
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Try to open with PIL to validate
        image = Image.open(BytesIO(image_data))
        
        logger.info(f"Image info - Format: {image.format}, Mode: {image.mode}, Size: {image.size}")
        
        return True, {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "data_size": len(image_data)
        }
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False, str(e)

# Initialize tesseract setup
try:
    tessdata_path = setup_tesseract()
    fast_mrz = FastMRZ(lang='mrz')
    logger.info("✓ FastMRZ initialized successfully")
except Exception as e:
    logger.error(f"✗ Error initializing FastMRZ: {e}")
    fast_mrz = None

class ImageBase64Request(BaseModel):
    base64_image: str
    ignore_parse: bool = False

class MRZTextRequest(BaseModel):
    mrz_text: str

@app.post("/extract")
def extract_mrz_from_base64(req: ImageBase64Request):
    if fast_mrz is None:
        raise HTTPException(status_code=503, detail="FastMRZ not initialized")
        
    try:
        logger.info(f"Received extraction request - ignore_parse: {req.ignore_parse}")
        logger.info(f"Base64 data length: {len(req.base64_image)}")
        
        # Basic validation
        if not req.base64_image or not req.base64_image.strip():
            raise HTTPException(status_code=400, detail="Empty base64 image data")
            
        if len(req.base64_image) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large")
        
        # Validate image
        is_valid, info = validate_base64_image(req.base64_image)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {info}")
        
        logger.info(f"Image validation passed: {info}")
        
        # Process with FastMRZ
        logger.info("Starting MRZ extraction...")
        result = fast_mrz.get_details(
            req.base64_image, 
            input_type="base64", 
            ignore_parse=req.ignore_parse
        )
        
        logger.info(f"MRZ extraction result: {result}")
        return JSONResponse(content=result)
        
    except ValueError as e:
        logger.error(f"ValueError in MRZ extraction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        logger.error(f"MRZ extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process MRZ: {str(e)}"
        )

@app.post("/validate")
def validate_mrz_text(req: MRZTextRequest):
    if fast_mrz is None:
        raise HTTPException(status_code=503, detail="FastMRZ not initialized")
        
    try:
        is_valid = fast_mrz.validate_mrz(req.mrz_text)
        return JSONResponse(content={"valid": is_valid})
    except Exception as e:
        logger.error(f"MRZ validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "FastMRZ API is running", "status": "ok"}

@app.get("/health")
async def health():
    if fast_mrz is None:
        raise HTTPException(status_code=503, detail="FastMRZ not initialized")
    
    tessdata_path = os.environ.get('TESSDATA_PREFIX', '')
    mrz_file = os.path.join(tessdata_path, 'mrz.traineddata')
    
    return {
        "status": "healthy",
        "tessdata_path": tessdata_path,
        "mrz_traineddata_exists": os.path.isfile(mrz_file),
        "mrz_traineddata_size": os.path.getsize(mrz_file) if os.path.isfile(mrz_file) else 0,
        "fastmrz_initialized": fast_mrz is not None
    }

@app.get("/tesseract-info")
async def tesseract_info():
    """Debug endpoint to check tesseract configuration"""
    info = {
        "tessdata_prefix_env": os.environ.get('TESSDATA_PREFIX'),
        "detected_tessdata_path": find_tesseract_data_path(),
        "available_languages": []
    }
    
    try:
        # Get tesseract version
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info["tesseract_version"] = result.stdout.split('\n')[0]
    except:
        info["tesseract_version"] = "Error getting version"
    
    try:
        result = subprocess.run(['tesseract', '--list-langs'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info["available_languages"] = result.stdout.strip().split('\n')[1:]  # Skip first line
        else:
            info["tesseract_error"] = result.stderr
    except Exception as e:
        info["available_languages"] = [f"Error getting languages: {str(e)}"]
    
    # Check if MRZ trained data exists
    tessdata_path = os.environ.get('TESSDATA_PREFIX')
    if tessdata_path:
        mrz_file = os.path.join(tessdata_path, 'mrz.traineddata')
        info["mrz_traineddata_exists"] = os.path.isfile(mrz_file)
        info["mrz_traineddata_path"] = mrz_file
        if os.path.isfile(mrz_file):
            info["mrz_traineddata_size"] = os.path.getsize(mrz_file)
    
    return info

@app.post("/debug-extract")
async def debug_extract(req: ImageBase64Request):
    """Debug endpoint that provides detailed extraction information"""
    if fast_mrz is None:
        raise HTTPException(status_code=503, detail="FastMRZ not initialized")
    
    debug_info = {
        "image_info": None,
        "tesseract_info": None,
        "extraction_result": None,
        "error": None
    }
    
    try:
        # Validate image
        is_valid, image_info = validate_base64_image(req.base64_image)
        debug_info["image_info"] = image_info if is_valid else {"error": image_info}
        
        if not is_valid:
            debug_info["error"] = f"Invalid image: {image_info}"
            return debug_info
        
        # Get tesseract info
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True, timeout=10)
            debug_info["tesseract_info"] = {
                "available_languages": result.stdout.strip().split('\n')[1:] if result.returncode == 0 else [],
                "mrz_available": 'mrz' in (result.stdout or ''),
                "tessdata_prefix": os.environ.get('TESSDATA_PREFIX')
            }
        except Exception as e:
            debug_info["tesseract_info"] = {"error": str(e)}
        
        # Try extraction
        result = fast_mrz.get_details(
            req.base64_image,
            input_type="base64",
            ignore_parse=req.ignore_parse
        )
        debug_info["extraction_result"] = result
        
    except Exception as e:
        debug_info["error"] = str(e)
        logger.error(f"Debug extraction failed: {e}", exc_info=True)
    
    return debug_info