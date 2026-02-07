# Bone Buddy Frontend

Upload a DICOM image and view the analysis result from the Bone Buddy API.

## Run

1. Start the API (from project root):
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

2. Install and start the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173 in your browser. Upload a DICOM file and click **Analyze**.

The frontend calls `POST http://localhost:8000/v1/analyze/dicom` and displays body part, fracture detection, bone age (for hand), and the generated report.
