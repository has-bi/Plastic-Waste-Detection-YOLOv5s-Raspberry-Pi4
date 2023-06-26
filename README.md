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

To train the YOLOv5s model effectively, the dataset was annotated using the Roboflow platform. Roboflow provides a user-friendly interface and annotation tools that facilitate the annotation process, saving time and effort.

The annotations include bounding boxes that specify the location and size of each plastic waste object within the images. These annotations are crucial for training the model to accurately detect and classify plastic waste objects.
