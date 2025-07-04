from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmrz import FastMRZ
from fastapi.responses import JSONResponse

app = FastAPI()
fast_mrz = FastMRZ()

class ImageBase64Request(BaseModel):
    base64_image: str
    ignore_parse: bool = False

class MRZTextRequest(BaseModel):
    mrz_text: str

@app.post("/extract")
def extract_mrz_from_base64(req: ImageBase64Request):
    try:
        result = fast_mrz.get_details(
            req.base64_image, input_type="base64", ignore_parse=req.ignore_parse
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
def validate_mrz_text(req: MRZTextRequest):
    try:
        is_valid = fast_mrz.validate_mrz(req.mrz_text)
        return JSONResponse(content={"valid": is_valid})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
