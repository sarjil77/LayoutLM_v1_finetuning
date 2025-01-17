from content.layout_lm_tutorial.layoutlm_preprocess import *
import pytesseract
from PIL import Image
import cv2
import json

from flask import Flask, request, render_template, jsonify
app = Flask(__name__)

def get_input():
    image_path = 'your folder pathcontent/data/testing_data/images/invoice_250-page-1.jpg'
    image = cv2.imread(image_path)
    return image

def iob_to_label(label):
  if label != 'O':
    return label[2:]
  else:
    return ""
  
def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

def get_response(num_labels, label_map, final_predictions):
    model_path='your folder pathcontent/layoutlm_invoices_May6_100epoch.pt'
    model=model_load(model_path,num_labels)
    image, words, boxes, actual_boxes = preprocess("your folder pathcontent/data/testing_data/images/invoice_250-page-1.jpg")

    word_level_predictions, final_boxes=convert_to_features(image, words, boxes, actual_boxes, model)
    for prediction, box in zip(word_level_predictions, final_boxes):
        predicted_label = iob_to_label(label_map[prediction]).upper()
        if predicted_label:
            x1, y1, x2, y2 = box
            # Crop image to the rectangle
            image = get_input()
            cropped_image = image[y1-10:y2+10, x1-10:x2+10]

            # Perform OCR on the cropped region
            text = pytesseract.image_to_string(cropped_image)
            
            # Print the extracted text
            if len(text)>2:
                # print(predicted_label, text)
                final_predictions[predicted_label].add(text.strip('\n\x0c'))
    
    return final_predictions

def json_data (response):
    for key, value in response.items():
        if isinstance(value, set):
            response[key] = list(value)

    # Convert dictionary to JSON string
    json_string = json.dumps(response)
    return json_string

@app.route('/invoice_response', methods=['POST'])
def invoice_response():
    labels = get_labels("your folder pathcontent/data/labels.txt")
    num_labels = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}
    invoice_labels = set([item[2:] for item in label_map.values() if len(item) > 1])
    final_predictions = {item: set() for item in invoice_labels}
    response=get_response(num_labels, label_map, final_predictions)
    json_response = json_data (response)
    return jsonify(json_response)
    
if __name__ == '__main__':
    app.run(debug=True)