import cv2
import torch
import time
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from yolo import attempt_load

class Deteksi:
    """
    Class for object detection using YOLOv5 model.

    Args:
        weights (str): Path to the YOLOv5 model file (.pt).
        labels (list): List of class labels.
        size (int): Input image size for inference (default: 416).
        confidence (float): Minimum confidence threshold for detection (default: 0.7).
        threshold (float): IoU threshold for non-maximum suppression (default: 0.3).
    """

    def __init__(self, weights, labels, size=416, confidence=0.7, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.labels = labels
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.net = attempt_load(weights, device=self.device)
            self.net.eval()
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
        image = cv2.resize(image, (self.size, self.size))
        image = F.to_tensor(image)
        image = image.unsqueeze(0)

        # Perform inference
        start = time.time()
        with torch.no_grad():
            outputs = self.net(image)
        end = time.time()
        inference_time = end - start
        # Process the outputs
        predictions = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]

                if scores.numel() == 0:
                    continue

                confidence, class_id = torch.max(scores, dim=0, keepdim=True)
                confidence = confidence.item()
                class_id = class_id.item()

                if confidence > self.confidence:
                    box = detection[:4] * torch.tensor([self.size, self.size, self.size, self.size]).view(4, 1)
                    box = box.int().tolist()
                    prediction = {
                        'box': box,
                        'confidence': confidence,
                        'class_id': class_id
                    }
                    predictions.append(prediction)

        # Perform non-maximum suppression
        predictions = self.nonMaxSuppression(predictions)

        return predictions

    def nonMaxSuppression(self, predictions):
        """
        Applies non-maximum suppression to filter out overlapping detections.

        Args:
            predictions (list): List of dictionaries representing the detected objects.

        Returns:
            list: List of dictionaries representing the filtered detections after non-maximum suppression.
        """
        if len(predictions) == 0:
            return []

        # Extract bounding box coordinates and confidence scores
        boxes = torch.tensor([pred['box'] for pred in predictions])
        confidence_scores = torch.tensor([pred['confidence'] for pred in predictions])

        # Compute the area of each bounding box
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Sort the predictions by confidence scores in descending order
        _, sorted_indices = confidence_scores.sort(descending=True)

        # Initialize an empty list to store the filtered predictions
        filtered_predictions = []

        while len(sorted_indices) > 0:
            # Get the prediction with the highest confidence score
            best_pred_index = sorted_indices[0]
            best_pred = predictions[best_pred_index]

            # Add the best prediction to the filtered predictions list
            filtered_predictions.append(best_pred)

            # Compute the intersection over union (IoU) between the best prediction and the rest of the predictions
            current_box = torch.tensor(best_pred['box']).float().unsqueeze(0)
            rest_of_boxes = boxes[sorted_indices[1:]].float()
            intersection_x1 = torch.max(current_box[:, 0], rest_of_boxes[:, 0])
            intersection_y1 = torch.max(current_box[:, 1], rest_of_boxes[:, 1])
            intersection_x2 = torch.min(current_box[:, 2], rest_of_boxes[:, 2])
            intersection_y2 = torch.min(current_box[:, 3], rest_of_boxes[:, 3])
            intersection_width = torch.clamp(intersection_x2 - intersection_x1, min=0)
            intersection_height = torch.clamp(intersection_y2 - intersection_y1, min=0)
            intersection_area = intersection_width * intersection_height

            current_box_area = areas[best_pred_index].unsqueeze(0)
            rest_of_areas = areas[sorted_indices[1:]].unsqueeze(1)

            iou = intersection_area / (current_box_area + rest_of_areas - intersection_area)

            # Remove predictions with IoU higher than the threshold
            sorted_indices = sorted_indices[1:][iou.squeeze() <= self.threshold]

        return filtered_predictions