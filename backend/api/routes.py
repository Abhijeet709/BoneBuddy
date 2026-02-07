from fastapi import APIRouter, UploadFile, File
from services.inference import run_full_inference

router = APIRouter(prefix="/v1")

@router.post("/analyze/dicom")
async def analyze_dicom(file: UploadFile = File(...)):
    result = await run_full_inference(file)
    return result
