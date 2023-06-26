# Plastic-Waste-Detection-YOLOv5
## Introduction
Welcome to the repository for the project "Plastic Waste Detection using YOLOv5s on Raspberry Pi 4B"! This project focuses on utilizing computer vision techniques to detect and classify plastic waste in real-time using the YOLOv5s object detection model, implemented on a Raspberry Pi 4B. The plastic waste detection classify the waste into PET, HDPE, PP and Non Recycleables.

## Project Higlights
- **YOLOv5s** : We employ the YOLOv5s (You Only Look Once version 5 small) object detection model as the core of our system. YOLOv5s offers a good balance between accuracy and speed, making it suitable for real-time applications on resource-constrained devices like the Raspberry Pi 4B.
- **Raspberry Pi 4B** : The Raspberry Pi 4B serves as the hardware platform for our project. Its compact size, low power consumption, and GPIO (General Purpose Input/Output) capabilities make it an ideal choice for edge computing and IoT applications.
- **Real-time Detection** :  Our implementation enables real-time detection and classification of plastic waste objects captured by the Webcam.

# Repository Structure
- Dataset Preparation
- Model Training
- Detection
 
# Dataset Preparation
The "Plastic Waste Detection using YOLOv5s on Raspberry Pi 4B" project utilizes a custom dataset consisting of 6000 images. These images were captured to encompass various types of plastic waste commonly found in the environment, including plastic bottles, plastic bags, plastic cups, cables, soap bottles, and styrofoam.
![Dataset-01](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/7a6f6584-e03b-46ab-b5e9-480218b84498)
To train the YOLOv5s model effectively, the dataset was annotated using the Roboflow platform. Roboflow provides a user-friendly interface and annotation tools that facilitate the annotation process, saving time and effort.

The annotations include bounding boxes that specify the location and size of each plastic waste object within the images. These annotations are crucial for training the model to accurately detect and classify plastic waste objects.
![image](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/4f53133b-268e-4948-b2bf-eb19e1123944)

# Model Training
The "Plastic Waste Detection using YOLOv5s on Raspberry Pi 4B" model has been trained using the YOLOv5s architecture implemented with the PyTorch framework. The choice of YOLOv5s is based on a balance between model accuracy and size, making it suitable for deployment on resource-constrained devices like the Raspberry Pi.

## Dataset Proportion
The dataset used for training the model has been divided into three main subsets: training, validation, and testing. The proportions of these subsets are as follows:
- **Training:** 70% of the dataset is allocated for training the YOLOv5s model. This subset is used to teach the model to detect and classify plastic waste objects accurate
- **Validation:** 20% of the dataset is set aside for validation purposes. This subset helps evaluate the model's performance during the training process, allowing for fine-tuning and parameter optimization.
- **Testing:** The remaining 10% of the dataset is dedicated to testing the trained model's performance. This subset provides an independent evaluation of the model's ability to generalize to unseen plastic waste scenarios.

## Training Process
To train the "Plastic Waste Detection using YOLOv5s on Raspberry Pi 4B" model with the specified parameters, you can follow the steps outlined below:
1. **Load the necessary framework and libraries:**
```
!git clone https://github.com/ultralytics/yolov5  # clone repository from ultralytics
%cd yolov5

%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```
2. **Assamble the dataset**
```
# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"
```
```
from roboflow import Roboflow
rf = Roboflow(api_key="Your Roboflow API KEY")
project = rf.workspace("workspace").project("project name")
dataset = project.version(1).download("yolov5")
```

3. **Train the model**

The training process involves feeding the annotated dataset to the YOLOv5s model and optimizing its parameters to minimize the detection errors. This process iterates over multiple epochs, with each epoch representing a complete pass through the entire training dataset.

During training, the model learns to identify various plastic waste objects by adjusting its internal weights. The loss function used guides the model to minimize the discrepancies between the predicted bounding boxes and the ground truth annotations.
```
!python train.py --img 416 --batch 16 --epochs 200 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

4. **Model Evaluation**

To assess the performance of the trained model, both the validation and testing subsets are used. The validation subset helps monitor the model's progress during training, while the testing subset provides an unbiased evaluation of its performance on unseen data.

Various evaluation metrics can be employed to measure the model's accuracy, such as precision, recall, and mean average precision (mAP). These metrics help quantify the model's ability to detect and classify plastic waste objects correctly.
```
%load_ext tensorboard
%tensorboard --logdir runs
```
After the evaluation, you can download the model in **/runs/train/exp** called **best.pt**

# Detection

The "Plastic Waste Detection using YOLOv5s on Raspberry Pi 4B" project includes a prototype setup using Blender to simulate the waste separation process. The prototype consists of various components, including servo motors, an LCD display, an IR sensor, a webcam, and four partitions in the bin representing different waste categories.

## Prototype Components
**1. Servo Motors:** The prototype features two servo motors. The first servo motor, located at the top level, is responsible for separating the waste to the right or left. The second servo motor, positioned at the lower level, separates the waste to the front or back.

**2. LCD Display:** A 16x2 LCD display is integrated into the prototype. It provides visual feedback by displaying the plastic waste category and the computational time for the detection process.

**3. IR Sensor:** An infrared (IR) sensor is utilized to detect the presence of waste at the base level. The sensor helps trigger the waste separation process when waste is detected.

**4. Webcam:** A webcam with 1080p resolution is used to capture the live video feed of the waste items for detection. The webcam provides input to the Raspberry Pi 4B, which processes the video feed using the YOLOv5s model.

**5. Bin with Partitions:** The bin in the prototype setup is divided into four partitions, representing different waste categories: PET, HDPE, PP, and Non-recyclable waste. These partitions facilitate the separation of waste based on its detected category.

![Proto-02](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/62b21e5d-14af-419d-9875-f93fd3a47a47)
![Proto-03](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/cb039098-bf59-4632-9764-5daf3dccb33c)

## Program 

This code implements a waste sorting system using a Raspberry Pi and computer vision. It utilizes YOLO object detection to identify and classify various types of plastic waste in real-time, including cables, plastic bags, plastic bottles, plastic cups, soap bottles, and styrofoam. The code initializes necessary modules and loads YOLO weights and labels. It captures video from a camera and processes each frame using YOLO for object detection. Detected objects are displayed with bounding boxes and labels. Based on the object class, corresponding actions are performed, such as displaying recycling messages and controlling servo motors for waste sorting. The code runs in a loop until the user exits and releases system resources. Overall, it enables efficient waste sorting using computer vision and Raspberry Pi.

**Deteksi.py**
```
import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
import time
import drivers
import yaml
from utils.general import non_max_suppression
from Fungsi import Deteksi

# Load YOLO weights and labels
print("Load bobot YOLO...")
yolo = Deteksi("./best.pt", ["cable", "plastic bag", "plastic bottle", "plastic cup", "soap bottle", "sterofoam"])

# Load class labels from data.yaml
classes = []
with open("data.yaml","r") as f:
	data = yaml.safe_load(f)
	if "names" in data:
    	classes = data["names"]

# Set parameters
yolo.size = int(640)
yolo.confidence = float(0.5)
Cable, PlasticBag, PlasticBottle, PlasticCup, SoapBottle, Sterofoam = 0, 0, 0, 0, 0, 0

# Initialize GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
display = drivers.Lcd()
GPIO.setup(38, GPIO.IN) #infrared
GPIO.setup(33, GPIO.OUT) #servo1
GPIO.setup(12, GPIO.OUT) #servo2
servo1_pwm=GPIO.PWM(33, 50)
servo2_pwm=GPIO.PWM(12, 50)
servo1_pwm.start(7.5)
time.sleep(1)
servo1_pwm.ChangeDutyCycle(0)
servo2_pwm.start(4.5)
time.sleep(1)
servo2_pwm.ChangeDutyCycle(0)
selesai_load = time.time()
infrared= GPIO.input(38)

# Display text on LCD
display.lcd_display_string("STAND", 1)
display.lcd_display_string("BY", 2)
time.sleep(2)
display.lcd_display_string("MENYALAKAN", 1)
display.lcd_display_string("KAMERA", 2)
video = cv.VideoCapture(0)
display.lcd_clear()

while True:
 	# Read frame from video capture
	ret, frame = video.read()
	frame = cv.flip(frame, 1)
	start = time.process_time()
	infared = 1
    
	if not ret:
    	continue
           	 
	predictions, inference_time = yolo.inference(frame)
	width = frame.shape[1]
	height = frame.shape[0]

	# Process each detection
	for prediction in predictions:
    	box= prediction['box']
    	confidence = prediction['confidence']
    	class_id = prediction['class_id']
   	 
   	 
    	x, y, w, h = box
    	cx = x + (w / 2)
    	cy = y + (h / 2)

	# Draw bounding box
    	color = (255, 0, 0)
    	cv.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), color, 2)
    	name = classes[class_id]
    	text = f"{name} ({round(confidence, 2)})" # Show the confidence score of predictions
    	cv.putText(frame, text, (int(x), int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
   	 
    	# Perform actions based on object class
    	if name == 'plastic bottle':
        	PlasticBottle += 1
        	if PlasticBottle > 5:
            	display.lcd_display_string("-Recycle-", 1)
            	time.sleep(1)
            	display.lcd_display_string("PET", 2)
            	servo1_pwm.ChangeDutyCycle(11.5)  # Servo1 Right 75 deg Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(7.5)  # Servo1 Neutral Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(9)  # Servo2 Right 75 deg Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(4.5)  # Servo2 Neutral Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(0)
            	servo2_pwm.ChangeDutyCycle(0)
            	display.lcd_clear()
            	end = time.process_time()
            	waktu = end - start
            	display.lcd_display_string("Waktu Komputasi:", 1)
            	display.lcd_display_string(str(waktu), 2)
            	time.sleep(4)
            	display.lcd_clear()
            	PlasticBottle = 0
            	break

    	elif name == 'plastic cup':
        	PlasticCup += 1
        	if PlasticCup > 5:
            	display.lcd_display_string("-Recycle-", 1)
            	time.sleep(1)
            	display.lcd_display_string("PP", 2)
            	servo1_pwm.ChangeDutyCycle(3.5)  # Servo1 Left 75 deg Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(7.5)  # Servo1 Neutral Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(9)  # Servo2 Left 75 deg Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(4.5)  # Servo2 Neutral Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(0)
            	servo2_pwm.ChangeDutyCycle(0)
            	display.lcd_clear()
            	end = time.process_time()
            	waktu = end - start
            	display.lcd_display_string("Waktu Komputasi:", 1)
            	display.lcd_display_string(str(waktu), 2)
            	time.sleep(4)
            	display.lcd_clear()
            	PlasticCup = 0
            	break

    	elif name == 'soap bottle':
        	SoapBottle += 1
        	if SoapBottle > 5:
            	display.lcd_display_string("-Recycle-", 1)
            	time.sleep(1)
            	display.lcd_display_string("HDPE", 2)
            	servo1_pwm.ChangeDutyCycle(3.5)  # Servo1 Left 75 deg Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(7.5)  # Servo1 Neutral Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(1.2)  # Servo2 Right 75 deg Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(4.5)  # Servo2 Neutral Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(0)
            	servo2_pwm.ChangeDutyCycle(0)
            	display.lcd_clear()
            	end = time.process_time()
            	waktu = end - start
            	display.lcd_display_string("Waktu Komputasi:", 1)
            	display.lcd_display_string(str(waktu), 2)
            	time.sleep(4)
            	display.lcd_clear()
            	SoapBottle = 0
            	break

    	elif name == ('cable'):
        	Cable += 1
        	if Cable > 5:
            	display.lcd_display_string("-Non Recycle-", 1)
            	time.sleep(1)
            	display.lcd_display_string("PVC",2)
            	servo1_pwm.ChangeDutyCycle(11.5)  # Servo1 Left 75 deg Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(7.5)  # Servo1 Neutral Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(1.2)  # Servo2 Left 75 deg Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(4.5)  # Servo2 Neutral Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(0)
            	servo2_pwm.ChangeDutyCycle(0)
            	display.lcd_clear()
            	end = time.process_time()
            	waktu = end - start
            	display.lcd_display_string("Waktu Komputasi:", 1)
            	display.lcd_display_string(str(waktu), 2)
            	time.sleep(4)
            	display.lcd_clear()
            	Cable = 0
            	break

    	elif name =='sterofoam':
        	Sterofoam += 1
        	if Sterofoam > 5:
            	display.lcd_display_string("-Non Recycle-", 1)
            	time.sleep(1)
            	display.lcd_display_string("PS",2)
            	servo1_pwm.ChangeDutyCycle(11.5)  # Servo1 Left 75 deg Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(7.5)  # Servo1 Neutral Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(1.2)  # Servo2 Left 75 deg Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(4.5)  # Servo2 Neutral Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(0)
            	servo2_pwm.ChangeDutyCycle(0)
            	display.lcd_clear()
            	end = time.process_time()
            	waktu = end - start
            	display.lcd_display_string("Waktu Komputasi:", 1)
            	display.lcd_display_string(str(waktu), 2)
            	time.sleep(4)
            	display.lcd_clear()
            	Sterofoam = 0
            	break

    	elif name =='plastic bag':
        	PlasticBag += 1
        	if PlasticBag > 5:
            	display.lcd_display_string("-Non Recycle-", 1)
            	time.sleep(1)
            	display.lcd_display_string("LDPE",2)
            	servo1_pwm.ChangeDutyCycle(11.5)  # Servo1 Left 75 deg Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(7.5)  # Servo1 Neutral Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(1.2)  # Servo2 Left 75 deg Position
            	time.sleep(4)
            	servo2_pwm.ChangeDutyCycle(4.5)  # Servo2 Neutral Position
            	time.sleep(4)
            	servo1_pwm.ChangeDutyCycle(0)
            	servo2_pwm.ChangeDutyCycle(0)
            	display.lcd_clear()
            	end = time.process_time()
            	waktu = end - start
            	display.lcd_display_string("Waktu Komputasi:", 1)
            	display.lcd_display_string(str(waktu), 2)
            	time.sleep(4)
            	display.lcd_clear()
            	PlasticBag = 0
            	break

   	 
	cv.imshow("PENGENALAN JENIS SAMPAH PLASTIK", frame)

	if cv.waitKey(1) & 0xFF == ord('q'):
    	break

# Release video capture and destroy windows
video.release()
cv.destroyAllWindow("preview")
GPIO.cleanup
```

The program, need the _Fungsi_ class, you can find it in **"Fungsi.py"**
```
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
```

## The Result of Detection
![Screenshot from 2023-06-22 10-43-20](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/5fdcba94-d3b4-4b74-a928-65cb6303d21b)
![Screenshot from 2023-06-22 10-53-37](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/a983f2c0-5680-4a99-8272-72a0174ebcc4)

![WhatsApp Image 2023-06-23 at 14 18 29](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/826e77ae-a661-45a2-9fa5-afe85e3c9df3)
![WhatsApp Image 2023-06-23 at 14 18 20](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5/assets/117572919/f64f9601-18eb-47d6-9eb9-b246eff97cde)

