from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
sys.path.append("./models")
from models.model import predict_malignant_benign

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Breast Cancer Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_path = "temp_image.jpg"  # Save the image temporarily
        image.save(image_path)

        # Provide dummy paths for the models
        mobilenet_model_path = "models/mobilenet_unfreeze_6_adamw.pth"
        mlp_model_path = "models/mlp_breast_cancer_model.h5"

        result = predict_malignant_benign(
            image_path,
            mobilenet_model_path,
            mlp_model_path,
            n_blocks=10
        )

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
