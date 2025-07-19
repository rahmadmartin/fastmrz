from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmrz import FastMRZ
from fastapi.responses import JSONResponse
import os
import subprocess
import sys
from pathlib import Path


app = FastAPI()
def find_tesseract_data_path():
    """
    Dynamically find the correct Tesseract data path and ensure MRZ trained data exists
    """
    possible_paths = [
        # Common Ubuntu/Debian paths
        "/usr/share/tesseract-ocr/5.00/tessdata/",
        "/usr/share/tesseract-ocr/4.00/tessdata/", 
        "/usr/share/tessdata/",
        "/usr/local/share/tessdata/",
        
        # Common paths on other systems
        "/opt/homebrew/share/tessdata/",  # macOS with Homebrew
        "/usr/local/Cellar/tesseract/*/share/tessdata/",  # macOS Homebrew
        "/Program Files/Tesseract-OCR/tessdata/",  # Windows
        
        # Try to get from tesseract command
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
    except:
        pass
    
    # Try to get from TESSDATA_PREFIX environment variable
    tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
    if tessdata_prefix:
        possible_paths.insert(0, tessdata_prefix)
    
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
                        return expanded_path
        else:
            if os.path.isdir(path):
                mrz_file = os.path.join(path, 'mrz.traineddata')
                if os.path.isfile(mrz_file):
                    return path
    
    return None

def download_mrz_traineddata(tessdata_path):
    """
    Download MRZ trained data if it doesn't exist
    """
    mrz_file = os.path.join(tessdata_path, 'mrz.traineddata')
    if os.path.isfile(mrz_file):
        return True
    
    print(f"MRZ trained data not found in {tessdata_path}")
    print("Attempting to download MRZ trained data...")
    
    try:
        import urllib.request
        url = "https://github.com/rahmadmartin/fastmrz/raw/refs/heads/main/tessdata/mrz.traineddata"
        
        # Create directory if it doesn't exist
        os.makedirs(tessdata_path, exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, mrz_file)
        
        if os.path.isfile(mrz_file):
            print(f"✓ MRZ trained data downloaded to {mrz_file}")
            return True
        else:
            print(f"✗ Failed to download MRZ trained data to {mrz_file}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading MRZ trained data: {e}")
        return False

def setup_tesseract():
    """
    Setup Tesseract with proper tessdata path
    """
    # Find tessdata path
    tessdata_path = find_tesseract_data_path()
    
    if not tessdata_path:
        print("Could not find tesseract tessdata directory. Trying common locations...")
        # Try to create in a common location
        common_paths = [
            "/usr/share/tesseract-ocr/4.00/tessdata/",
            "/usr/share/tessdata/",
            "/usr/local/share/tessdata/"
        ]
        
        for path in common_paths:
            try:
                os.makedirs(path, exist_ok=True)
                if download_mrz_traineddata(path):
                    tessdata_path = path
                    break
            except PermissionError:
                continue
    
    if not tessdata_path:
        raise RuntimeError("Could not find or create tesseract tessdata directory")
    
    # Ensure MRZ trained data exists
    if not download_mrz_traineddata(tessdata_path):
        raise RuntimeError(f"Could not download MRZ trained data to {tessdata_path}")
    
    # Set environment variable
    os.environ['TESSDATA_PREFIX'] = tessdata_path
    
    print(f"✓ Tesseract tessdata path set to: {tessdata_path}")
    return tessdata_path

# Initialize tesseract setup
try:
    tessdata_path = setup_tesseract()
    fast_mrz = FastMRZ(lang='mrz')
    print("✓ FastMRZ initialized successfully")
except Exception as e:
    print(f"✗ Error initializing FastMRZ: {e}")
    fast_mrz = None
class ImageBase64Request(BaseModel):
    base64_image: str
    ignore_parse: bool = False

class MRZTextRequest(BaseModel):
    mrz_text: str

@app.post("/extract")
def extract_mrz_from_base64(req: ImageBase64Request):
    try:
        print(f"Received base64 data (first 100 chars): {req.base64_image}...")
        # Basic validation
        if not req.base64_image or not req.base64_image.strip():
            raise HTTPException(status_code=400, detail="Empty base64 image data")
            
        if len(req.base64_image) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large")
            
        result = fast_mrz.get_details(
            req.base64_image, 
            input_type="base64", 
            ignore_parse=req.ignore_parse
        )
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        # Log the full error for debugging
        # logger.error(f"MRZ extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to process MRZ - the image may not contain a readable MRZ zone"
        )

@app.post("/validate")
def validate_mrz_text(req: MRZTextRequest):
    try:
        is_valid = fast_mrz.validate_mrz(req.mrz_text)
        return JSONResponse(content={"valid": is_valid})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "FastMRZ API is running", "status": "ok"}

@app.get("/health")
async def health():
    if fast_mrz is None:
        raise HTTPException(status_code=503, detail="FastMRZ not initialized")
    
    return {
        "status": "healthy",
        "tessdata_path": os.environ.get('TESSDATA_PREFIX'),
        "mrz_traineddata_exists": os.path.isfile(
            os.path.join(os.environ.get('TESSDATA_PREFIX', ''), 'mrz.traineddata')
        )
    }