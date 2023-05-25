import cv2
import os
from segmentation import Segment
from ocr import get_text_list

from datetime import datetime

def save_images(images, dir, file_name, img_type):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: Creating directory ' + dir)
     
    for idx in range(len(images)):
        now = datetime.now()
        dt_string = now.strftime('%Y%m%d%H%M%S')
        target_file = file_name + '_' + str(idx+1) + '_' + dt_string + '.' + img_type
        cv2.imwrite(str(os.path.join(dir, target_file)), images[idx])
    return dir

def get_ocr_result(user_name, image, dir='',img_type='png'):
    file_name = user_name + '_book_img'
    segment = Segment()
    cropped_images = segment.get_masked_images(image, is_cropped=True)
    path = save_images(cropped_images, dir, file_name, img_type)
    ocr_text_list = get_text_list(path, img_type)
    return ocr_text_list


user_name = '7777'
image = cv2.imread('./test_13.jpeg')
target_dir = './book_img/segment/'
ocr_result = get_ocr_result(user_name, image, target_dir, 'jpeg')
print(ocr_result)