import os
import tempfile
from dicom.loader import load_dicom
from dicom.preprocess import preprocess_xray
from models.body_part import BodyPartModel
from models.fracture import FractureModel
from models.bone_age import BoneAgeModel
from services.llm_report import generate_report

body_model = BodyPartModel()
fracture_model = FractureModel()
bone_age_model = BoneAgeModel()

async def run_full_inference(uploaded_file):
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(await uploaded_file.read())
            path = f.name
        image, dicom = load_dicom(path)

        processed = preprocess_xray(image)

        body_part, body_conf = body_model.predict(processed)
        fracture, frac_conf = fracture_model.predict(processed)

        bone_age = None
        if body_part == "hand":
            bone_age = bone_age_model.predict(processed)

        report = generate_report(body_part, fracture, bone_age)

        return {
            "body_part": body_part,
            "body_part_confidence": body_conf,
            "fracture_detected": fracture,
            "fracture_confidence": frac_conf,
            "bone_age_months": bone_age,
            "report": report,
            "warning": "Not for clinical use"
        }
    finally:
        if path and os.path.exists(path):
            os.remove(path)
