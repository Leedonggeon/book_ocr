import os
import cv2
import json, requests
import uuid
import time

api_url = ''
secret_key = ''

def request_ocr(files, img_type):
    request_json = {'images': [{'format': img_type,
                                'name': 'demo'
                               }],
                    'requestId': str(uuid.uuid4()),
                    'version': 'V2',
                    'timestamp': int(round(time.time() * 1000))
                   }
 
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    
    headers = {
    'X-OCR-SECRET': secret_key,
    }
    
    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
    result = response.json()
    return result

def process_result(result):
    segmented_text_list = []
    for field in result['images'][0]['fields']:
        segmented_text_list.append(field['inferText'])
    return segmented_text_list

def get_text(image_dir, img_type):
    files = [('file', open(image_dir, 'rb'))]
    result = request_ocr(files, img_type)
    text = process_result(result)
    return text

# Input : image_path
# Output : Texts List
def get_text_list(dir_path, img_type = 'jpg'):
    ocr_text_list = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            path = os.path.join(dir_path, path)
            text = get_text(path, img_type)
            ocr_text_list.append(text)
    return ocr_text_list