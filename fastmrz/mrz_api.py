from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmrz import FastMRZ
from fastapi.responses import JSONResponse

app = FastAPI()
fast_mrz = FastMRZ(lang='ocrb')  # Use both mrz and eng traineddata
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
        logger.error(f"MRZ extraction failed: {str(e)}", exc_info=True)
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

@app.get("/health")
def health_check():
    return {"status": "healthy"}