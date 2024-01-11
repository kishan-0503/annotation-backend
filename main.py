from typing import Dict
from fastapi import FastAPI, status, Body, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from pdf2image import convert_from_bytes
from base64 import encodebytes
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pytesseract
import cv2
import requests
import io
import base64
from zipfile import ZipFile

from schema import Coordinate, TrainingCoordinate, OCRRequest


app: FastAPI = FastAPI(
    name="xemi-annotation-api",
    docs_url = "/",
    debug = True, 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/ocr/from-image-url/')
async def read_image_from_url(data: Coordinate):
    '''
    Send post request with Coordinate and image url
    API will read and return text from given Coordinate of image. 
    '''
    content = requests.get(data.image_url).content
    array = np.asarray(bytearray(content), dtype=np.uint8)

    image = cv2.imdecode(array, -1) 
    crop_img = image[data.y: data.y+data.h, data.x: data.x+data.w]
    ocr_df = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DATAFRAME)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"ocr_text": ' '.join(ocr_df['text'])},
    )


@app.get('/ocr/document-data/{document_id}/')
async def get_document_data(document_id: int):
    '''
    Get document data
    '''
    url = 'https://stageapi.xemi.io/api/v1/annotation/' + '{}'.format(document_id)
    res = requests.request('GET', url)

    if not res.status_code == 200:
        return JSONResponse({}, status_code=status.HTTP_409_CONFLICT,)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=res.json(),
    )


@app.post('/ocr/approved-data/')
async def process_approved_data(data: Dict = Body(...)):
    '''
    process_approved_data
    call webhook from here
    '''
    url = 'https://stageapi.xemi.io/internal_webhook/'
    headers = {'Content-Type': 'application/json'}
    res = requests.request('POST', url, headers=headers, data=data)

    if not res.status_code == 200:
        return JSONResponse({}, status_code=status.HTTP_409_CONFLICT,)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=res.json(),
    )


@app.get('/training/document-data/{document_id}/')
async def get_training_document_data(document_id: int, file_url: str):
    '''
    Get training document data
    '''
    url = 'https://stageapi.xemi.io/api/v1/momenttext/document/' + '{}'.format(document_id)
    res = requests.request('GET', url)
    if not res.status_code == 200:
        return JSONResponse({}, status_code=status.HTTP_409_CONFLICT,)

    encoded_pages = []
    zipfile = ZipFile(io.BytesIO(requests.get(file_url).content), 'r')

    for file_name in zipfile.namelist():

        if ".png" in file_name:
            document = Image.open(io.BytesIO(zipfile.read(file_name)))
            byte_array = io.BytesIO()
            document.save(byte_array, format='PNG') 
            encoded_page = encodebytes(byte_array.getvalue()).decode('ascii') 
            encoded_pages.append(encoded_page)

    # Check for the presence of the "annotated_response" key
    if "annotated_response" in res.json()["data"]:
        # Check if the value of the "annotated_response" key is not null
        if res.json()["data"]["annotated_response"] is not None:
            # Return the "annotated_response" value if it exists and is not null
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={'data': res.json()["data"]["annotated_response"]["data"]["result"], 'document_pages': encoded_pages}
            )
        else:
            # Return the "momenttext_response" value if the "annotated_response" value is null
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={'data': res.json()["data"]["momenttext_response"]["data"]["result"], 'document_pages': encoded_pages}
            )
    else:
        # Return the "momenttext_response" value if "annotated_response" does not exist
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'data': res.json()["data"]["momenttext_response"]["data"]["result"], 'document_pages': encoded_pages}
        )


import traceback
import sys

@app.post('/ocr/from-image-bytes/')
async def read_image_from_bytes(data: TrainingCoordinate):
    '''
    Send post request with Coordinate and image url
    API will read and return text from given Coordinate of image. 
    '''
    try:
        from PIL import Image
        stream = io.BytesIO(data.image)
        img = Image.open(stream)
        print(img)

        # array = np.asarray(bytearray(data.image), dtype=np.uint8)

        # image = cv2.imdecode(array, -1)
        
        # crop_img = image[data.y: data.y+data.h, data.x: data.x+data.w]
        # ocr_df = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DATAFRAME)
        # ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        # ocr_df = ocr_df.dropna().reset_index(drop=True)
        # print(ocr_df['text'])
    
    except Exception as e:
        print(traceback.format_exc())
        print(sys.exc_info()[2])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"ocr_text": 'This is dummy test text'},
        )


@app.post("/ocr/from-image-file/")
async def read_image_from_file(
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    file: UploadFile = File(...),
):
    contents = file.file.read()
    array = np.asarray(bytearray(contents), dtype=np.uint8)
    image = cv2.imdecode(array, -1) 
    crop_img = image[y: y+h, x: x+w]
    ocr_df = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DATAFRAME)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"ocr_text": ' '.join(ocr_df['text'])},
    )

@app.post("/ocr/from-base64-image/")
async def read_image_from_base64(request: OCRRequest):
    """
    API to get ocr data from base64 image
    """
    try:
        image_data = base64.b64decode(request.base64_image)
        array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(array, -1)
        if image is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Failed to decode the image."},
            )
            # raise HTTPException(status_code=400, detail="Failed to decode the image.")

        crop_img = image[request.y: request.y + request.h, request.x: request.x + request.w]

        ocr_df = pytesseract.image_to_data(crop_img, output_type=pytesseract.Output.DATAFRAME)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"ocr_text": ' '.join(ocr_df['text'])},
        )

    except Exception as e:
        return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": e},
            )
        # raise HTTPException(status_code=500, detail=f"An error occurred: {e}")