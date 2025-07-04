# FastMRZ: Machine Readable Zone Extraction Toolkit

![FastMRZ Logo](https://raw.githubusercontent.com/sivakumar-mahalingam/fastmrz/main/docs/FastMRZ.png)

FastMRZ is a high-performance Python library for extracting and validating Machine Readable Zone (MRZ) data from travel documents like passports and ID cards. Built with computer vision and OCR technologies, it supports multiple input formats and provides structured output.

## Key Features

✨ **Multi-format Input Support**
- Process images (JPG/PNG), base64 strings, MRZ text, or NumPy arrays
- Coming soon: PDF document support

🔍 **Advanced Extraction**
- Custom contour detection for MRZ region localization
- ONNX-optimized models for fast processing
- Tesseract OCR with specialized MRZ training

✅ **Data Validation**
- Comprehensive checksum verification
- MRZ format validation (ICAO Doc 9303 compliant)
- Field-level data integrity checks

📊 **Flexible Output**
- Raw MRZ text extraction
- Structured JSON with parsed fields
- Configurable output options

## Installation

### Prerequisites
- Tesseract OCR 5.0+ ([Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))
- Python 3.8+

### Install Package
```bash
pip install fastmrz
Configure Tesseract
Download mrz.traineddata

Copy to your Tesseract tessdata directory

Quick Start
python
from fastmrz import FastMRZ
import json

# Initialize with default settings
processor = FastMRZ()

# Process a document image
results = processor.get_details("passport.jpg")

# Output as formatted JSON
print(json.dumps(results, indent=2))
API Reference
FastMRZ Class
Parameters:

tesseract_path (str): Custom path to Tesseract executable

model_path (str): Path to custom ONNX model

Methods:

get_details(input, ignore_parse=False, include_checkdigit=True): Process input and return MRZ data

validate_mrz(mrz_text): Validate MRZ checksums and format

get_mrz_text(input): Extract raw MRZ text only

Supported MRZ Formats
FastMRZ supports all ICAO-standard MRZ formats:

Type	Documents	Lines	Characters
TD1	ID Cards	3	30
TD2	Old ID Cards	2	36
TD3	Passports	2	44
MRVA	Visas (Type A)	2	44
MRVB	Visas (Type B)	2	36
Performance Benchmarks
Operation	Avg Time (ms)
Image Processing	120
MRZ Extraction	85
Full Pipeline	205
*Tested on Intel i7-1185G7 @ 3.0GHz with 16GB RAM*

Roadmap
PDF document support

Bulk processing mode

Face image extraction

Enhanced image preprocessing

MRZ generation capability

Contributing
We welcome contributions! Please see our Contribution Guidelines for details.

License
AGPL-3.0 - See LICENSE for more information.

Who's Using FastMRZ?
Add your organization here!

API Service Option
FastMRZ also provides a ready-to-deploy REST API:

bash
pip install fastmrz[api]
uvicorn fastmrz.api:app --host 0.0.0.0 --port 8000
API endpoints:

POST /extract - Process MRZ from base64 image

POST /validate - Validate MRZ text

Swagger docs available at /docs when running locally.