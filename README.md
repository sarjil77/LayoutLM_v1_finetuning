# LayoutLM v1 Fine-tuning for Invoice Processing

A comprehensive project for fine-tuning Microsoft's LayoutLM v1 model on invoice document understanding tasks. This project enables extraction of key information from invoice documents including invoice numbers, dates, client information, seller details, and more through token classification.

## 🎯 Project Overview

This project implements document understanding using LayoutLM v1, a pre-trained multimodal (text + layout) model that combines textual and spatial information for form understanding. The model is fine-tuned on invoice datasets to extract structured information from scanned invoice documents.

### Key Features

- **Token Classification**: Extract invoice entities like invoice number, date, client info, seller details
- **Multi-modal Processing**: Leverages both text content and spatial layout information
- **OCR Integration**: Uses Tesseract OCR for text extraction from document images
- **REST API**: Flask-based API endpoints for invoice processing
- **Multiple Datasets**: Support for both FUNSD and custom invoice datasets
- **Inference Pipeline**: Complete workflow from image input to structured JSON output

## 📊 Supported Entity Types

The model extracts the following invoice entities:

- **Invoice Information**: Invoice number, invoice date
- **Client Details**: Client name, client address, client tax ID
- **Seller Information**: Seller name, seller address, seller tax ID, IBAN
- **Other**: Miscellaneous text elements

## 🏗️ Project Structure

```
LayoutLM_v1_finetuning/
├── README.md                           # This file
├── LayoutLM_invoices.ipynb            # Main training notebook for invoice dataset
├── LayoutLM_FUNSD.ipynb               # Training notebook for FUNSD dataset
├── Inferencing.ipynb                  # Inference examples and testing
├── all_json_prepare.ipynb             # Data preparation from OCR to JSON format
├── prepare_input_files.ipynb          # Input file preprocessing
├── checkImbalance.ipynb               # Dataset analysis and class imbalance check
├── invoice_response_json.ipynb        # JSON response formatting examples
├── Invoice_May9.ipynb                 # Training experiments from May 9
├── PaperSummary_LayoutLM.ipynb        # LayoutLM paper summary and analysis
├── preprocess.py                      # Data preprocessing utilities
├── invoice_API.py                     # Flask API for invoice processing
├── invoice_image_API.py               # Flask API with image upload support
├── training_loss_May11.txt            # Training loss logs
├── 1000_Invoice_data_may_9/           # Invoice dataset (XML annotations)
├── content/                           # Main content directory
│   ├── data/                         # Training and testing datasets
│   ├── transformers/                 # Hugging Face transformers library
│   ├── unilm/                        # UniLM repository with LayoutLM
│   ├── layout_lm_tutorial/           # Tutorial and preprocessing utilities
│   ├── postman_data/                 # API testing data
│   └── sample_data/                  # Sample images and annotations
├── json_prepare/                      # JSON data preparation scripts
├── issueIn/                          # Issue tracking and debugging files
└── layout_lm_tutorial/               # Additional tutorial resources
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Tesseract OCR
- Required Python packages (see Installation)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd LayoutLM_v1_finetuning
```

2. **Install system dependencies:**
```bash
# Install Tesseract OCR
sudo apt install tesseract-ocr

# For other systems, follow Tesseract installation guide
```

3. **Set up Python environment:**
```bash
# Create virtual environment
python -m venv layoutlm_env
source layoutlm_env/bin/activate  # On Windows: layoutlm_env\Scripts\activate

# Install required packages
pip install torch torchvision transformers
pip install pytesseract pillow opencv-python
pip install flask pandas numpy seqeval
pip install jupyter notebook
```

4. **Clone required repositories:**
```bash
# Clone UniLM repository (contains LayoutLM implementation)
git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git content/unilm

# Clone Transformers repository
git clone https://github.com/huggingface/transformers.git content/transformers

# Install packages
cd content/unilm/layoutlm && pip install .
cd ../../transformers && pip install .
```

### Dataset Setup

1. **For FUNSD Dataset:**
```bash
# Download FUNSD dataset
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip && mv dataset content/data && rm -rf dataset.zip __MACOSX
```

2. **For Invoice Dataset:**
   - Place your invoice images in `content/data/training_data/images/` and `content/data/testing_data/images/`
   - Place corresponding JSON annotations in `content/data/training_data/annotations/` and `content/data/testing_data/annotations/`

## 🎓 Usage

### 1. Data Preprocessing

Run the preprocessing to convert annotations to LayoutLM format:

```bash
# For training data
python content/unilm/layoutlm/examples/seq_labeling/preprocess.py \
    --data_dir content/data/training_data/annotations \
    --data_split train \
    --output_dir content/data \
    --model_name_or_path microsoft/layoutlm-base-uncased \
    --max_len 510

# For testing data
python content/unilm/layoutlm/examples/seq_labeling/preprocess.py \
    --data_dir content/data/testing_data/annotations \
    --data_split test \
    --output_dir content/data \
    --model_name_or_path microsoft/layoutlm-base-uncased \
    --max_len 510
```

### 2. Generate Labels File

```bash
# Extract unique labels from training data
cat content/data/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > content/data/labels.txt
```

### 3. Training

Open and run the training notebook:

```bash
jupyter notebook LayoutLM_invoices.ipynb
```

Key training parameters:
- **Learning Rate**: 5e-5
- **Batch Size**: 2
- **Epochs**: 5 (adjustable)
- **Max Sequence Length**: 512
- **Model**: microsoft/layoutlm-base-uncased

### 4. Inference

For inference on new documents:

```bash
jupyter notebook Inferencing.ipynb
```

Or use the standalone inference script with trained model.

### 5. API Deployment

Start the Flask API server:

```bash
# For basic API (fixed image path)
python invoice_API.py

# For image upload API
python invoice_image_API.py
```

API endpoints:
- `POST /invoice_response` - Process invoice and return extracted information

Example API usage:
```bash
# Using curl with image upload
curl -X POST -F "image=@path/to/invoice.jpg" http://localhost:5000/invoice_response
```

## 📋 Data Format

### Input Format
- **Images**: JPG/PNG format invoice documents
- **Annotations**: JSON format with bounding boxes and labels

Example annotation structure:
```json
{
  "form": [
    {
      "id": 1,
      "label": "invoice_number",
      "box": [x1, y1, x2, y2],
      "words": [
        {
          "text": "INV-001",
          "box": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

### Output Format
```json
{
  "InvoiceInfo": {
    "InvoiceNo": "INV-001",
    "DateOfIssue": "2023-05-01"
  },
  "Seller": {
    "Name": "Company ABC",
    "TaxId": "123456789",
    "Address": "123 Business St",
    "IBAN": "DE89370400440532013000"
  },
  "Client": {
    "Name": "Client XYZ",
    "TaxId": "987654321",
    "Address": "456 Client Ave"
  }
}
```

## 🔧 Configuration

### Model Configuration
- **Base Model**: microsoft/layoutlm-base-uncased
- **Task**: Token Classification
- **Labels**: BIOES tagging scheme (Begin, Inside, Outside, End, Single)

### Training Configuration
Update paths in notebooks and scripts:
- Replace `"path to your data"` with actual data paths
- Update model save/load paths
- Adjust GPU device settings (`cuda:0`, `cuda:1`, etc.)

## 📊 Model Performance

The model uses the following evaluation metrics:
- **Precision**: Token-level precision for each entity type
- **Recall**: Token-level recall for each entity type
- **F1-Score**: Harmonic mean of precision and recall
- **Loss**: Cross-entropy loss during training

Training logs are saved in `training_loss_May11.txt`.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Issues**:
   - Reduce batch size in training notebooks
   - Use gradient accumulation if needed

2. **Path Errors**:
   - Update all hardcoded paths in notebooks and scripts
   - Ensure data directories exist before training

3. **OCR Issues**:
   - Verify Tesseract installation: `tesseract --version`
   - Check image quality and format

4. **Model Loading Errors**:
   - Ensure model file exists at specified path
   - Check model architecture compatibility

### Performance Optimization

- Use mixed precision training for faster training
- Increase batch size if GPU memory allows
- Use data parallel training for multiple GPUs

## 📚 Additional Resources

- **LayoutLM Paper**: [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
- **FUNSD Dataset**: [Form Understanding in Noisy Scanned Documents](https://guillaumejaume.github.io/FUNSD/)
- **Hugging Face LayoutLM**: [Model Hub](https://huggingface.co/microsoft/layoutlm-base-uncased)

## 🔄 Workflow Summary

1. **Data Preparation**: Convert invoice images and annotations to LayoutLM format
2. **Preprocessing**: Tokenize text and normalize bounding boxes
3. **Training**: Fine-tune LayoutLM on invoice dataset
4. **Evaluation**: Assess model performance on test set
5. **Inference**: Extract information from new invoice documents
6. **API Deployment**: Serve model through REST API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:
- Open an issue on GitHub
- Check existing notebooks for examples
- Review the troubleshooting section

---

**Note**: This project is for research and educational purposes. Update file paths and configurations according to your specific environment before running.