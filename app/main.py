from io import BytesIO

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.inference.predict import predict_image


app = FastAPI(title="Waste Classification API")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    result = predict_image(image, model_type)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "filename": file.filename,
            "model_used": result["model_used"],
        },
    )