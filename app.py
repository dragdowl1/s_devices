import os
import ssl
import torch
import uvicorn
import logging
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from warnings import filterwarnings, simplefilter
from fastapi import FastAPI, Request, File, UploadFile


filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

if not os.path.exists('logs'):
    os.mkdir('logs')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler()
file_handler = logging.FileHandler('logs/api.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


from predict import process_image
models = {}

app = FastAPI()
@app.on_event("startup")
async def startup_event():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt', force_reload = True)
    # model = torch.hub.load(repo_or_dir= 'yolov5', model = 'custom', path = 'best.pt', source = 'local', force_reload = True)
    model.conf = 0.35
    models["1"] = model

@app.get("/")
async def root():
    return {"message": "setup complete"}


@app.post("/object_detect")
async def image_detect(request: Request,
                      file: UploadFile = File(...)):

    if request.method == "POST":
        json_result = []
        model = models["1"]
        try:

            image = Image.open(file.file)
            json_results = process_image(image, model)


            return JSONResponse({"data": json_results,
                                 "message": "object detected successfully",
                                 "errors": None,
                                 "status": 200},
                                status_code=200)
        except Exception as error:
            return JSONResponse({"message": "object detection failed",
                                 "errors": "error",
                                 "status": 400},
                                status_code=400)
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
