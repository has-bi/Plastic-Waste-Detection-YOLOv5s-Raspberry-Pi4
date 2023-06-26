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
