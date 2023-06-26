import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from models.yolo import attempt_load
from utils.general import non_max_suppression

class Deteksi:
	def __init__(self, weights, labels, size=640, confidence=0.5, threshold=0.3):
    	self.confidence = confidence
    	self.threshold = threshold
    	self.size = size
    	self.labels = labels
    	try:
        	self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        	self.model = attempt_load(weights)
        	self.model.eval()
    	except:
        	raise ValueError("Failed to load model file (.pt)")

	def inference(self, image):
    	"""
    	Preprocesses the input image and performs object detection.

    	Args:
        	image (numpy.ndarray): Input image as a NumPy array.

    	Returns:
        	list: List of dictionaries representing the detected objects, each with the following keys:
            	- 'box' (list): Bounding box coordinates [x, y, width, height].
            	- 'confidence' (float): Confidence score of the detection.
            	- 'class_id' (int): Index of the predicted class label.
    	"""
    	# Preprocess the image
    	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    	img = Image.fromarray(image)
    	img = img.resize((self.size, self.size), Image.BICUBIC)
    	img = F.to_tensor(img).unsqueeze(0)

    	# Perform inference
    	start = time.time()
    	with torch.no_grad():
        	img = img.to(self.device)
        	pred = self.model(img)[0]
    	end = time.time()
    	inference_time = end - start

    	# Process the outputs
    	pred = non_max_suppression(pred, self.confidence, self.threshold)[0]

    	# Prepare the results
    	predictions = []
    	if pred is not None and len(pred) > 0:
        	# Rescale bounding boxes to original image size
        	pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()

        	# Iterate over detections and create prediction dictionary
        	for *box, conf, class_id in pred:
            	x1, y1, x2, y2 = box
            	box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            	confidence = float(conf)
            	class_id = int(class_id)

            	prediction = {
                	'box': box,
                	'confidence': confidence,
                	'class_id': class_id
            	}
            	predictions.append(prediction)

    	return predictions, inference_time

def scale_coords(img_shape, coords, img0_shape):
	# Rescale coordinates to original image size
	gain = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])
	pad_x = (img_shape[1] - img0_shape[1] * gain) / 2
	pad_y = (img_shape[0] - img0_shape[0] * gain) / 2
	coords[:, [0, 2]] -= pad_x
	coords[:, [1, 3]] -= pad_y
	coords[:, :4] /= gain
	clip_coords(coords, img0_shape)
	return coords.round()

def clip_coords(coords, img_shape):
	# Clip bounding coordinates to image shape
	coords[:, 0].clamp_(0, img_shape[1])  # x1
	coords[:, 1].clamp_(0, img_shape[0])  # y1
	coords[:, 2].clamp_(0, img_shape[1])  # x2
	coords[:, 3].clamp_(0, img_shape[0])  # y2
	return coords
