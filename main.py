from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from loguru import logger
import cv2
import math
import base64
import time
import json

ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
app = FastAPI()

class Image(BaseModel):
    images: List[str]

def sort_result(polygons):
    polygons = polygons.reshape(-1, 4, 2)
    indices = np.lexsort((polygons[:, 0, 0], polygons[:, 0, 1]))
    sorted_polygons = polygons[indices]
    sorted_polygons = sorted_polygons.reshape(-1, 8)
    return sorted_polygons

@app.post("/predict/ocr_system")
async def prediction(items:Image):
    try:
        try:
            results = []
            for img_b64 in items.images:
                result_single = []
                img_binary = base64.b64decode(img_b64)
                buffer = np.frombuffer(img_binary,dtype=np.uint8)
                img = cv2.imdecode(buffer,flags=cv2.IMREAD_COLOR)
                det_result = ocr_detection(img)
                logger.debug(f"det_result: {det_result}")
                det_result = det_result['polygons'] 
                det_result = sort_result(det_result)
                for i in range(det_result.shape[0]):
                    start_time = time.time()
                    pts = order_point(det_result[i])
                    logger.debug(f"pts: {pts}")
                    image_crop = crop_image(img, pts)
                    output = ocr_recognition(image_crop)
                    end_time = time.time()
                    each_text = {
                        "confidence": end_time-start_time,
                        "text": output['text'][0],
                        "text_region": pts.tolist()
                    }
                    result_single.append(each_text)
                results.append(result_single)
                
            final_result = {
                "msg": "", 
                "results": results,
                "status": "000"
            }
            logger.info(results)
            return final_result
        except:
            logger.debug("No text detected, try using rec directly.")
            start_time = time.time()
            output = ocr_recognition(img)
            logger.debug(f"output: \n{output}")
            end_time = time.time()
            each_text = {
                "confidence": end_time-start_time,
                "text": output['text'][0],
                "text_region": []
            }
            result_single.append(each_text)
            results.append(result_single)
            final_result = {
                "msg": "", 
                "results": results,
                "status": "000"
            }
            logger.info(results)
            return final_result
    except Exception as e:
        logger.error(e)
        res = {
            "msg": str(e), 
            "results": [],
            "status": "000"
        }
        return res
            
                


def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst

def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points


img_path = 'default.jpg'
image_full = cv2.imread(img_path)
det_result = ocr_detection(image_full)
det_result = det_result['polygons'] 
for i in range(det_result.shape[0]):
    pts = order_point(det_result[i])
    image_crop = crop_image(image_full, pts)
    result = ocr_recognition(image_crop)
#     print("box: %s" % ','.join([str(e) for e in list(pts.reshape(-1))]))
#     print("text: %s" % result['text'])

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

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
        img_path = 'default.jpg'
        image_full = cv2.imread(img_path)
        det_result = ocr_detection(image_full)
        det_result = det_result['polygons'] 
        for i in range(det_result.shape[0]):
            pts = order_point(det_result[i])
            image_crop = crop_image(image_full, pts)
            result = ocr_recognition(image_crop)
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

