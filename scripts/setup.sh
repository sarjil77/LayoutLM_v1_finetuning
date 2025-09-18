#!/bin/bash
# Setup script for LayoutLM v1 Fine-tuning Project

echo "🚀 Setting up LayoutLM v1 Fine-tuning Project..."

# Create directories if they don't exist
mkdir -p data/raw/invoices/{images,annotations}
mkdir -p data/raw/funsd
mkdir -p data/sample/uploads
mkdir -p models/trained

echo "📁 Created required directories"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Clone dependencies if they don't exist
if [ ! -d "dependencies/unilm" ]; then
    echo "📥 Cloning UniLM repository..."
    git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git dependencies/unilm
fi

if [ ! -d "dependencies/transformers" ]; then
    echo "📥 Cloning Transformers repository..."
    git clone https://github.com/huggingface/transformers.git dependencies/transformers
fi

# Install packages
echo "🔧 Installing LayoutLM and Transformers..."
cd dependencies/unilm/layoutlm && pip install . && cd ../../..
cd dependencies/transformers && pip install . && cd ..

echo "✅ Setup complete!"
echo ""
echo "📖 Next steps:"
echo "1. Place your data in data/raw/invoices/ or download FUNSD dataset"
echo "2. Run preprocessing: python src/preprocessing/data_preprocessor.py"
echo "3. Start training: jupyter notebook notebooks/training/layoutlm_invoices_training.ipynb"
echo "4. Test API: python src/api/image_api.py"
echo ""
echo "📚 Check README.md for detailed instructions"