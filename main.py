from fastapi import FastAPI , status , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
# from TTS.api import TTS

from time import strftime
from time import time
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate_onnx import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from urllib.parse import urlencode
from loguru import logger



import requests
import json
import os, sys
import base64
import os

tts_service = os.getenv("TTS_SERVER")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

pic_path ="./face.jpg"
facerender_batch_size = 10
sadtalker_paths = init_path("./checkpoints", os.path.join("/home/SadTalker", 'src/config'), "256", False, "full")

preprocess_model = CropAndExtract(sadtalker_paths, "cuda")
audio_to_coeff = Audio2Coeff(sadtalker_paths, "cuda")
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, "cuda")

app = FastAPI()

origins = ["*"]
 
app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

class Words(BaseModel):
    words: str


@app.post("/pipeline")
async def predict_image(items:Words):
    save_dir = os.path.join("/home/SadTalker/results", strftime("%Y_%m_%d_%H.%M.%S"))
    """
    从语音服务器获取语音内容
    """
    try:
        params = {
            "text": items.words,
            "speaker_id": "p376",
            "language_id": "en"
        }
        
        # Construct the full URL with parameters
        full_url = f"{tts_service}?{urlencode(params)}"
        
        logger.debug(f"Sending request to TTS server: URL={full_url}")

        response = requests.get(full_url)
        response.raise_for_status()
        logger.debug(f"Received response from TTS server: Status={response.status_code}")
    
        if response.headers.get('Content-Type') == 'audio/wav':
            audio_data = response.content
        else:
            # If it's not direct audio, try to parse as JSON
            audio_base64 = response.json().get('audio')
            if not audio_base64:
                raise ValueError("No audio data in response")
            audio_data = base64.b64decode(audio_base64)
  
        audio_path = "/home/SadTalker/output.wav"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_data)
    except Exception as e:
        errors = str(e)
        mod_errors = errors.replace('"', '**').replace("'", '**')
        logger.error(mod_errors)
        message = {
            "err_no": "400",
            "err_msg": mod_errors
            }
        json_data = json.dumps(message)
        json_data = json_data.replace("'", '"')
        return json_data

    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, "full", source_image_flag=True)
    ref_eyeblink_coeff_path=None
    ref_pose_coeff_path=None
    batch = get_data(first_coeff_path, audio_path, "cuda", ref_eyeblink_coeff_path, still=True)
    coeff_path = audio_to_coeff.generate(batch, save_dir, 0, ref_pose_coeff_path)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                facerender_batch_size, None, None, None,
                                expression_scale=1, still_mode=True, preprocess="full")
    video_path = animate_from_coeff.generate_deploy(data, save_dir, pic_path, crop_info, \
                                enhancer="gfpgan", background_enhancer=None, preprocess="crop")
    with open(video_path, "rb") as file:
            video_data = base64.b64encode(file.read()).decode("utf-8")
    response = {
            "video_base64": video_data
        }
        
    return response
    
    
        
@app.get("/health")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

@app.get("/health/inference")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)