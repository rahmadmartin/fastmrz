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
import cv2
import numpy as np
import pytesseract
import json
import re
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enum for document types
class DocumentType(str, Enum):
    passport = "passport"
    ktp = "ktp"

# KTP Information class
class KTPInformation:
    def __init__(self):
        self.nik = ""
        self.nama = ""
        self.tempat_lahir = ""
        self.tanggal_lahir = ""
        self.jenis_kelamin = ""
        self.golongan_darah = ""
        self.alamat = ""
        self.rt = ""
        self.rw = ""
        self.kelurahan_atau_desa = ""
        self.kecamatan = ""
        self.agama = ""
        self.status_perkawinan = ""
        self.pekerjaan = ""
        self.kewarganegaraan = ""
        self.berlaku_hingga = "SEUMUR HIDUP"

# KTP OCR class
class KTPOCR:
    def __init__(self, image_array):
        self.image = image_array
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.th, self.threshed = cv2.threshold(self.gray, 127, 255, cv2.THRESH_TRUNC)
        self.result = KTPInformation()
        self.master_process()

    def process(self):
        try:
            raw_extracted_text = pytesseract.image_to_string(self.threshed, lang="ind")
            return raw_extracted_text
        except Exception as e:
            logger.error(f"Error in pytesseract processing: {e}")
            # Fallback to English if Indonesian is not available
            try:
                raw_extracted_text = pytesseract.image_to_string(self.threshed, lang="eng")
                return raw_extracted_text
            except Exception as e2:
                logger.error(f"Error in fallback pytesseract processing: {e2}")
                raise e2

    def word_to_number_converter(self, word):
        word_dict = {
            '|': "1"
        }
        res = ""
        for letter in word:
            if letter in word_dict:
                res += word_dict[letter]
            else:
                res += letter
        return res

    def nik_extract(self, word):
        word_dict = {
            'b': "6",
            'e': "2",
            'S': "5",
            'O': "0",
            'I': "1",
            'l': "1"
        }
        res = ""
        for letter in word:
            if letter in word_dict:
                res += word_dict[letter]
            else:
                res += letter
        return res
    
    def extract(self, extracted_result):
        # Replace newlines for logging (can't use backslash in f-string)
        log_text = extracted_result.replace('\n', ' -- ')
        logger.info(f"Extracted text: {log_text}")
        
        for word in extracted_result.split("\n"):
            word = word.strip()
            if not word:
                continue
                
            # NIK extraction - improved to handle various formats
            if "NIK" in word.upper():
                # Handle cases like "NIK 3 1302076504950001" or "NIK : 1302076504950001"
                nik_value = word.upper().replace("NIK", "").replace(":", "").strip()
                # Remove all non-digit characters and take exactly 16 digits
                nik_digits = re.sub(r'[^\d]', '', nik_value)
                if len(nik_digits) == 16:
                    self.result.nik = nik_digits
                elif len(nik_digits) > 16:
                    # If there are extra digits (like the '3' before the NIK), take last 16
                    self.result.nik = nik_digits[-16:]
                logger.info(f"Raw NIK value: '{word}' -> Extracted: '{self.result.nik}'")
                continue

            # Name extraction
            if "Nama" in word and ":" in word:
                parts = word.split(':')
                if len(parts) > 1:
                    self.result.nama = parts[-1].strip()
                continue

            # Birth place and date extraction
            if "Tempat" in word and "Lahir" in word:
                parts = word.split(':')
                if len(parts) > 1:
                    birth_info = parts[-1].strip()
                    # Look for date pattern
                    date_match = re.search(r"([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4})", birth_info)
                    if date_match:
                        self.result.tanggal_lahir = date_match.group(0)
                        self.result.tempat_lahir = birth_info.replace(self.result.tanggal_lahir, '').strip()
                    else:
                        self.result.tempat_lahir = birth_info
                continue

            # Gender and blood type extraction
            if 'Darah' in word or 'Kelamin' in word:
                # Extract gender
                gender_match = re.search(r"(LAKI-LAKI|LAKI|LELAKI|PEREMPUAN)", word.upper())
                if gender_match:
                    self.result.jenis_kelamin = gender_match.group(0)
                
                # Extract blood type
                blood_match = re.search(r"\b(O|A|B|AB)\b", word.upper())
                if blood_match:
                    self.result.golongan_darah = blood_match.group(0)
                else:
                    self.result.golongan_darah = '-'
                continue
            
            # Address extraction
            if 'Alamat' in word:
                self.result.alamat = self.word_to_number_converter(word).replace("Alamat", "").strip()
                continue
                
            if 'NO.' in word and self.result.alamat:
                self.result.alamat += ' ' + word.strip()
                continue
            
            # Sub-district extraction
            if "Kecamatan" in word and ":" in word:
                parts = word.split(':')
                if len(parts) > 1:
                    self.result.kecamatan = parts[1].strip()
                continue
            
            # Village extraction
            if "Desa" in word or "Kelurahan" in word:
                # Remove the word "Desa" or "Kelurahan" and extract the name
                clean_word = re.sub(r'(Desa|Kelurahan)', '', word, flags=re.IGNORECASE).strip()
                if ":" in clean_word:
                    parts = clean_word.split(':')
                    if len(parts) > 1:
                        self.result.kelurahan_atau_desa = parts[1].strip()
                continue
            
            # Citizenship extraction
            if 'Kewarganegaraan' in word and ":" in word:
                parts = word.split(':')
                if len(parts) > 1:
                    self.result.kewarganegaraan = parts[1].strip()
                continue
            
            # Occupation extraction
            if 'Pekerjaan' in word:
                clean_word = word.replace('Pekerjaan', '').strip()
                if ":" in clean_word:
                    parts = clean_word.split(':')
                    if len(parts) > 1:
                        self.result.pekerjaan = parts[1].strip()
                else:
                    self.result.pekerjaan = clean_word.strip()
                continue
            
            # Religion extraction
            if 'Agama' in word:
                clean_word = word.replace('Agama', '').strip()
                if ":" in clean_word:
                    parts = clean_word.split(':')
                    if len(parts) > 1:
                        self.result.agama = parts[1].strip()
                else:
                    self.result.agama = clean_word.strip()
                continue
            
            # Marital status extraction
            if 'Perkawinan' in word and ":" in word:
                parts = word.split(':')
                if len(parts) > 1:
                    self.result.status_perkawinan = parts[1].strip()
                continue
            
            # RT/RW extraction
            if "RT" in word.upper() and "RW" in word.upper():
                # Look for patterns like "RT 001/RW 002" or "001/002"
                rt_rw_match = re.search(r"(\d+)[/\s]*(\d+)", word)
                if rt_rw_match:
                    self.result.rt = rt_rw_match.group(1).zfill(3)
                    self.result.rw = rt_rw_match.group(2).zfill(3)
                continue

    def master_process(self):
        try:
            raw_text = self.process()
            self.extract(raw_text)
        except Exception as e:
            logger.error(f"Error in KTP processing: {e}")
            raise e

    def to_dict(self):
        return {
            "nik": self.result.nik,
            "nama": self.result.nama,
            "tempat_lahir": self.result.tempat_lahir,
            "tanggal_lahir": self.result.tanggal_lahir,
            "jenis_kelamin": self.result.jenis_kelamin,
            "golongan_darah": self.result.golongan_darah,
            "alamat": self.result.alamat,
            "rt": self.result.rt,
            "rw": self.result.rw,
            "kelurahan_atau_desa": self.result.kelurahan_atau_desa,
            "kecamatan": self.result.kecamatan,
            "agama": self.result.agama,
            "status_perkawinan": self.result.status_perkawinan,
            "pekerjaan": self.result.pekerjaan,
            "kewarganegaraan": self.result.kewarganegaraan,
            "berlaku_hingga": self.result.berlaku_hingga
        }

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

def find_tesseract_executable():
    """
    Find tesseract executable in common locations
    """
    possible_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/opt/homebrew/bin/tesseract',
        '/snap/bin/tesseract'
    ]
    
    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.info(f"Found tesseract at: {path}")
            return path
    
    # Try to find using which command if available
    try:
        result = subprocess.run(['which', 'tesseract'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            logger.info(f"Found tesseract using 'which': {path}")
            return path
    except:
        pass
    
    return None

def setup_tesseract():
    """
    Setup Tesseract with proper tessdata path
    """
    # Find tesseract executable
    tesseract_path = find_tesseract_executable()
    
    if not tesseract_path:
        logger.error("Tesseract executable not found in common locations")
        raise RuntimeError("Tesseract executable not found")
    
    # Check if tesseract works
    try:
        result = subprocess.run([tesseract_path, '--version'], 
                              capture_output=True, text=True, timeout=10)
        logger.info(f"Tesseract version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
        
        # Store the tesseract path for later use
        os.environ['TESSERACT_CMD'] = tesseract_path
        
    except Exception as e:
        logger.error(f"Tesseract not working: {e}")
        raise RuntimeError(f"Tesseract found at {tesseract_path} but not working: {e}")
    
    # Find tessdata path (we know from logs it's at /usr/share/tesseract-ocr/5/tessdata/)
    tessdata_path = find_tesseract_data_path()
    
    if not tessdata_path:
        # From the logs, we know MRZ data exists at these locations
        known_good_paths = [
            "/usr/share/tesseract-ocr/5/tessdata/",
            "/usr/share/tesseract-ocr/4.00/tessdata/"
        ]
        
        for path in known_good_paths:
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'mrz.traineddata')):
                tessdata_path = path
                logger.info(f"Using known good tessdata path: {tessdata_path}")
                break
    
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
        tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')
        result = subprocess.run([tesseract_cmd, '--list-langs'], 
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

def base64_to_cv2_image(base64_string):
    """
    Convert base64 string to OpenCV image format
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
        
    except Exception as e:
        logger.error(f"Error converting base64 to CV2 image: {e}")
        raise e

# Initialize tesseract setup
fast_mrz = None
tessdata_path = None

def initialize_fastmrz():
    """Initialize FastMRZ with proper error handling"""
    global fast_mrz, tessdata_path
    
    if fast_mrz is not None:
        return True
        
    try:
        logger.info("Starting FastMRZ initialization...")
        
        # Find and check tesseract
        tesseract_path = find_tesseract_executable()
        if not tesseract_path:
            logger.error("Tesseract executable not found")
            return False
            
        try:
            result = subprocess.run([tesseract_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            logger.info(f"Tesseract found at {tesseract_path}: {result.stdout.split()[1] if result.stdout else 'Unknown version'}")
            os.environ['TESSERACT_CMD'] = tesseract_path
        except Exception as e:
            logger.error(f"Error checking tesseract at {tesseract_path}: {e}")
            return False
        
        # Setup tesseract
        tessdata_path = setup_tesseract()
        logger.info(f"Tessdata path configured: {tessdata_path}")
        
        # Initialize FastMRZ
        fast_mrz = FastMRZ(lang='mrz')
        logger.info("✓ FastMRZ initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error initializing FastMRZ: {e}", exc_info=True)
        fast_mrz = None
        return False

# Try to initialize on startup, but don't fail if it doesn't work
initialization_success = initialize_fastmrz()

class ImageBase64Request(BaseModel):
    base64_image: str
    type: DocumentType = DocumentType.passport  # Default to passport for backward compatibility
    ignore_parse: bool = False

class MRZTextRequest(BaseModel):
    mrz_text: str

@app.post("/extract")
def extract_document(req: ImageBase64Request):
    try:
        logger.info(f"Received extraction request - type: {req.type}, ignore_parse: {req.ignore_parse}")
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
        
        if req.type == DocumentType.passport:
            # Handle MRZ extraction
            if fast_mrz is None:
                logger.info("FastMRZ not initialized, attempting initialization...")
                if not initialize_fastmrz():
                    raise HTTPException(status_code=503, detail="FastMRZ not initialized and initialization failed")
            
            logger.info("Starting MRZ extraction...")
            result = fast_mrz.get_details(
                req.base64_image, 
                input_type="base64", 
                ignore_parse=req.ignore_parse
            )
            logger.info(f"MRZ extraction result: {result}")
            return JSONResponse(content=result)
            
        elif req.type == DocumentType.ktp:
            # Handle KTP extraction
            logger.info("Starting KTP extraction...")
            
            # Convert base64 to OpenCV image
            cv2_image = base64_to_cv2_image(req.base64_image)
            
            # Process with KTP OCR
            ktp_processor = KTPOCR(cv2_image)
            result = ktp_processor.to_dict()
            
            logger.info(f"KTP extraction result: {result}")
            return JSONResponse(content={
                "success": True,
                "data": result,
                "document_type": "ktp"
            })
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported document type: {req.type}")
        
    except ValueError as e:
        logger.error(f"ValueError in document extraction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        logger.error(f"Document extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process document: {str(e)}"
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
    return {"message": "FastMRZ API with KTP support is running", "status": "ok"}

@app.get("/health")
async def health():
    tessdata_path = os.environ.get('TESSDATA_PREFIX', '')
    mrz_file = os.path.join(tessdata_path, 'mrz.traineddata') if tessdata_path else ''
    
    # Check if tesseract languages are available
    tesseract_langs = []
    try:
        result = subprocess.run(['tesseract', '--list-langs'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            tesseract_langs = result.stdout.strip().split('\n')[1:]
    except:
        pass
    
    health_data = {
        "status": "healthy" if fast_mrz is not None else "partial",
        "tessdata_path": tessdata_path,
        "mrz_traineddata_exists": os.path.isfile(mrz_file) if mrz_file else False,
        "mrz_traineddata_size": os.path.getsize(mrz_file) if mrz_file and os.path.isfile(mrz_file) else 0,
        "fastmrz_initialized": fast_mrz is not None,
        "initialization_attempted": initialization_success,
        "available_languages": tesseract_langs,
        "supported_document_types": ["passport", "ktp"],
        "ktp_support": True
    }
    
    # If FastMRZ is not initialized, try to reinitialize
    if fast_mrz is None:
        logger.info("Attempting to reinitialize FastMRZ...")
        if initialize_fastmrz():
            health_data["status"] = "healthy"
            health_data["fastmrz_initialized"] = True
            health_data["reinitialized"] = True
        else:
            # Return health info even if FastMRZ is not working
            health_data["error"] = "FastMRZ initialization failed - passport extraction unavailable"
    
    return health_data

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

@app.get("/startup-logs")
async def startup_logs():
    """Get detailed startup information for debugging"""
    logs = []
    
    # Check tesseract installation
    tesseract_path = find_tesseract_executable()
    if tesseract_path:
        try:
            result = subprocess.run([tesseract_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            logs.append(f"✓ Tesseract found at {tesseract_path}: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
        except Exception as e:
            logs.append(f"✗ Error with tesseract at {tesseract_path}: {e}")
    else:
        logs.append("✗ Tesseract executable not found in common locations")
    
    # Check tessdata paths
    possible_paths = [
        "/usr/share/tesseract-ocr/5.00/tessdata/",
        "/usr/share/tesseract-ocr/4.00/tessdata/", 
        "/usr/share/tessdata/",
        "/usr/local/share/tessdata/",
        "/usr/share/tesseract-ocr/tessdata/",
    ]
    
    for path in possible_paths:
        exists = os.path.isdir(path)
        mrz_exists = os.path.isfile(os.path.join(path, 'mrz.traineddata')) if exists else False
        logs.append(f"Path {path}: dir_exists={exists}, mrz_exists={mrz_exists}")
    
    # Check environment
    tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
    logs.append(f"TESSDATA_PREFIX env var: {tessdata_prefix}")
    
    # Check available languages
    tesseract_cmd = find_tesseract_executable()
    if tesseract_cmd:
        try:
            result = subprocess.run([tesseract_cmd, '--list-langs'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                langs = result.stdout.strip().split('\n')[1:]
                logs.append(f"Available languages: {langs}")
                logs.append(f"MRZ available: {'mrz' in langs}")
                logs.append(f"Indonesian available: {'ind' in langs}")
                logs.append(f"English available: {'eng' in langs}")
            else:
                logs.append(f"Error listing languages: {result.stderr}")
        except Exception as e:
            logs.append(f"Error checking languages: {e}")
    else:
        logs.append("Cannot check languages - tesseract not found")
    
    # FastMRZ status
    logs.append(f"FastMRZ initialized: {fast_mrz is not None}")
    logs.append(f"Initialization attempted: {initialization_success}")
    
    # KTP support status
    logs.append("KTP support: Available (OpenCV + pytesseract)")
    logs.append("Supported document types: passport, ktp")
    
    return {"logs": logs}

@app.post("/debug-extract")
async def debug_extract(req: ImageBase64Request):
    """Debug endpoint that provides detailed extraction information"""
    debug_info = {
        "image_info": None,
        "tesseract_info": None,
        "extraction_result": None,
        "error": None,
        "document_type": req.type
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
                "ind_available": 'ind' in (result.stdout or ''),
                "eng_available": 'eng' in (result.stdout or ''),
                "tessdata_prefix": os.environ.get('TESSDATA_PREFIX')
            }
        except Exception as e:
            debug_info["tesseract_info"] = {"error": str(e)}
        
        # Try extraction based on document type
        if req.type == DocumentType.passport:
            if fast_mrz is None:
                if not initialize_fastmrz():
                    debug_info["error"] = "FastMRZ not initialized"
                    return debug_info
            
            result = fast_mrz.get_details(
                req.base64_image,
                input_type="base64",
                ignore_parse=req.ignore_parse
            )
            debug_info["extraction_result"] = result
            
        elif req.type == DocumentType.ktp:
            # Convert base64 to OpenCV image for debugging
            cv2_image = base64_to_cv2_image(req.base64_image)
            debug_info["cv2_image_shape"] = cv2_image.shape
            
            # Process with KTP OCR
            ktp_processor = KTPOCR(cv2_image)
            result = ktp_processor.to_dict()
            debug_info["extraction_result"] = {
                "success": True,
                "data": result,
                "document_type": "ktp"
            }
        
    except Exception as e:
        debug_info["error"] = str(e)
        logger.error(f"Debug extraction failed: {e}", exc_info=True)
    
    return debug_info