
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
import cv2
'''
fig, ax = plt.subplots(figsize=(18, 8))
ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
'''
class Segment():
    def __init__(self) -> None:
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        
        # for cpu
        cfg.MODEL.DEVICE = 'cpu'
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # for cpu
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
    
    # Input : Entire Image
    # Output : Masked Image

    def mask_image(self, idx, out, image, is_cropped):
        mask = out.pred_masks[idx]
        mask = np.asarray(mask, dtype='uint8')
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        if is_cropped:
            box = out.pred_boxes[idx].tensor.numpy()[0]
            masked_image = masked_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        return masked_image

    # Input : cv2.imread('***.jpg')
    #         numpy.ndarray
    # Output : List[numpy.ndarray_1, numpy.ndarray_2, ..., numpy.ndarray_n]
    def get_masked_images(self, image, is_cropped=False):
        outputs = self.predictor(image)
        outputs = outputs["instances"]#.to["cpu"]
        masked_image_list = []
        for idx in range(len(outputs.pred_masks)):
            masked_image = self.mask_image(idx, outputs, image, is_cropped)
            masked_image_list.append(masked_image)
        return masked_image_list