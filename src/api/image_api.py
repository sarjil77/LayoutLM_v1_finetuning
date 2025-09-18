from content.layout_lm_tutorial.layoutlm_preprocess import *
import pytesseract
# from PIL import Image
import cv2
import numpy as np  
import json
import os
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)

# def get_input(image_path):
#     image = cv2.imread(image_path)
#     return image

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

def get_response(image_path, num_labels, label_map, final_predictions):
    model_path='your folder path/Models/layoutlm_invoices_May6_100epoch.pt'
    model=model_load(model_path,num_labels)
    image, words, boxes, actual_boxes = preprocess(image_path)

    word_level_predictions, final_boxes=convert_to_features(image, words, boxes, actual_boxes, model)
    for prediction, box in zip(word_level_predictions, final_boxes):
        predicted_label = iob_to_label(label_map[prediction]).upper()
        if predicted_label:
            x1, y1, x2, y2 = box
            image=np.array(image)
            cropped_image = image[y1-10:y2+10, x1-10:x2+10]

            # Perform OCR on the cropped region
            text = pytesseract.image_to_string(cropped_image)
            
            # Print the extracted text
            if len(text)>2:
                # print(predicted_label, text)
                final_predictions[predicted_label].add(text.strip('\n\x0c'))
    
    return final_predictions

def json_data (response):
    response = {key: list(value) for key, value in response.items()}
    # Restructure the data
    restructured_data = {
        "InvoiceInfo": {
            "InvoiceNo": response["INVOICE_NUMBER"][0] if response["INVOICE_NUMBER"] else "",
            "DateOfIssue": response["INVOICE_DATE"][0] if response["INVOICE_DATE"] else ""
        },
        "Seller": {
            "Name": response["SELLER_NAME"][0] if response["SELLER_NAME"] else "",
            "TaxId": response["SELLER_TAX_ID"][0] if response["SELLER_TAX_ID"] else "",
            "Address": " ".join(response["SELLER_ADDRESS"]) if response["SELLER_ADDRESS"] else "",
            "IBAN": response["IBAN"][0] if response["IBAN"] else ""
        },
        "Client": {
            "Name": " ".join(response["CLIENT_NAME"]) if response["CLIENT_NAME"] else "",
            "TaxId": response["CLIENT_TAX_ID"][0] if response["CLIENT_TAX_ID"] else "",
            "Address": " ".join(response["CLIENT_ADDRESS"]) if response["CLIENT_ADDRESS"] else ""
        },
        "file_name": "".join(response["file_name"]) if response["file_name"] else ""
    }
    # Convert restructured data to JSON format
    restructured_json = json.dumps(restructured_data, indent=2)
    return restructured_json

@app.route('/invoice_response', methods=['POST'])
def invoice_response():
    postman_image=request.files['image']
    image_name=postman_image.filename
    if image_name:
        image_path=os.path.join('your folder path/content/postman_data',image_name)
        postman_image.save(image_path)
    else:
        print('No file from Postman')
    
    labels = get_labels("your folder path/content/data/labels.txt")
    num_labels = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}
    invoice_labels = set([item[2:] for item in label_map.values() if len(item) > 1])
    final_predictions = {item: set() for item in invoice_labels}
    response=get_response(image_path, num_labels, label_map, final_predictions)
    response.update({'file_name': image_name})
    json_response = json_data (response)
    return jsonify(json_response)
    
if __name__ == '__main__':
    app.run(debug=True)