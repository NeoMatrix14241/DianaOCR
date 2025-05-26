# Qwen 2.5VL Local OCR Setup

A simple Python script for batch OCR processing using Qwen 2.5VL model with CPU support and automatic GPU detection.

## Features

- **Multi-threading**: Utilizes all CPU cores for maximum performance
- **GPU Support**: Automatically detects and uses GPU if available, falls back to CPU
- **Batch Processing**: Processes entire folder structures maintaining hierarchy
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Error Handling**: Robust error handling with detailed logs
- **Flexible Input**: Supports various image formats (TIFF, PNG, JPG, etc.)

## Setup

### 1. Virtual Environment (Already Set Up)
Your virtual environment is already created. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download Model (First Run Only)
The model will be automatically downloaded from Hugging Face on first run. This may take some time depending on your internet connection.

## Usage

### Quick Start
```powershell
# Create test folder structure
python example_usage.py

# Run OCR on your images
python batch_ocr.py input output
```

### Command Line Options
```powershell
# Basic usage
python batch_ocr.py input_folder output_folder

# Specify device explicitly
python batch_ocr.py input_folder output_folder --device cpu

# Custom number of workers
python batch_ocr.py input_folder output_folder --workers 8

# Use different model
python batch_ocr.py input_folder output_folder --model "Qwen/Qwen2-VL-2B-Instruct"
```

### Folder Structure
Place your images in this structure:
```
input/
├── folder1/
│   ├── subfolder1/
│   │   ├── image1.tif
│   │   └── image2.tif
│   └── subfolder2/
│       ├── image1.tif
│       └── image2.tif
└── folder2/
    ├── image1.tif
    └── image2.tif
```

Output will maintain the same structure:
```
output/
├── folder1/
│   ├── subfolder1/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── subfolder2/
│       ├── image1.txt
│       └── image2.txt
└── folder2/
    ├── image1.txt
    └── image2.txt
```

## Configuration

### CPU Optimization
The script automatically:
- Uses all available CPU cores
- Sets optimal PyTorch thread count
- Uses float32 precision for CPU compatibility

### GPU Detection
- Automatically detects CUDA availability
- Falls back to CPU if no GPU found
- Uses optimal precision (bfloat16) for GPU

### Performance Tips
1. **CPU Usage**: For CPU-only processing, consider using a smaller model like `Qwen/Qwen2-VL-2B-Instruct`
2. **Memory**: Monitor RAM usage; reduce `max_workers` if running out of memory
3. **Batch Size**: Process images in smaller batches if experiencing memory issues

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face Hub access
   - Try manually downloading the model

2. **Out of Memory**
   - Reduce number of workers: `--workers 2`
   - Use smaller model: `--model "Qwen/Qwen2-VL-2B-Instruct"`
   - Close other applications

3. **Slow Performance on CPU**
   - Ensure all CPU cores are being used
   - Consider using GPU if available
   - Use smaller model for faster processing

### Logs
Check `ocr_processing.log` for detailed processing information and error messages.

## Future GUI Integration

The current script is designed for easy GUI integration. Key components that can be adapted:

- `QwenVLOCRProcessor`: Core OCR functionality
- `BatchOCRManager`: Batch processing logic
- Progress callbacks and status updates
- Configurable parameters (device, workers, model)

## Model Options

Available Qwen 2.5VL models (in order of size/performance):
- `Qwen/Qwen2-VL-2B-Instruct` (Fastest, least accurate)
- `Qwen/Qwen2-VL-7B-Instruct` (Balanced, default)
- `Qwen/Qwen2-VL-72B-Instruct` (Slowest, most accurate)

Choose based on your speed vs. accuracy requirements.
