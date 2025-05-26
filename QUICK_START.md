# Qwen 2.5VL Local OCR - Quick Start Guide

## ✅ Setup Complete!

Your Qwen 2.5VL OCR environment is ready for batch processing. Here's everything you need to know:

## 📁 Folder Structure Created

```
Qwen2.5-VL-72B/
├── input/                    ← Place your images here
│   ├── folder1/
│   │   ├── subfolder1/      ★ (PDF name)
│   │   └── subfolder2/      ★ (PDF name)
│   └── folder2/             ★ (PDF name)
├── output/                   ← OCR results appear here
├── batch_ocr.py             ← Main OCR script
├── demo_test.py             ← Test model loading
├── run_ocr.bat              ← Windows quick launcher
└── README.md                ← Detailed documentation
```

## 🚀 Quick Usage

### Option 1: Use the Windows Batch File (Easiest)
```cmd
# Double-click run_ocr.bat
# Or from command line:
run_ocr.bat
```

### Option 2: Command Line
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run OCR on all images in input folder
python batch_ocr.py input output

# Test model loading first (recommended)
python demo_test.py
```

## 🖼️ Supported Image Formats
- `.tif` and `.tiff` (primary target)
- `.png`, `.jpg`, `.jpeg`, `.bmp`

## ⚙️ Configuration Options

### CPU Optimization (Current Setup)
- **CPU Cores**: 12 (automatically detected)
- **Threading**: Optimized for all cores
- **Model**: Will use CPU-optimized settings
- **Memory**: Uses float32 precision for CPU compatibility

### Command Line Options
```powershell
# Use specific number of workers
python batch_ocr.py input output --workers 8

# Force CPU usage (default for your setup)
python batch_ocr.py input output --device cpu

# Use smaller/faster model
python batch_ocr.py input output --model "Qwen/Qwen2-VL-2B-Instruct"
```

## 📊 Performance Expectations

### First Run
- Model download: 5-15 minutes (one-time)
- Model loading: 30-60 seconds

### Subsequent Runs
- Model loading: 10-30 seconds
- Processing speed: ~30-60 seconds per image (CPU)

### Memory Usage
- Model: ~4-8 GB RAM
- Processing: Additional 2-4 GB RAM
- Total recommended: 8+ GB RAM

## 🔧 Troubleshooting

### Common Issues

1. **"Model not found" error**
   ```powershell
   # Check internet connection and try:
   python demo_test.py
   ```

2. **Out of memory**
   ```powershell
   # Reduce workers:
   python batch_ocr.py input output --workers 2
   ```

3. **Slow processing**
   ```powershell
   # Use smaller model:
   python batch_ocr.py input output --model "Qwen/Qwen2-VL-2B-Instruct"
   ```

### Log Files
- Check `ocr_processing.log` for detailed error information
- Processing results saved in `output/processing_results.json`

## 📝 Output Format

For each image, you'll get:
- `image_name.txt` - Extracted text
- `processing_results.json` - Detailed processing statistics

Example:
```
input/folder1/subfolder1/document.tif
→ output/folder1/subfolder1/document.txt
```

## 🎯 Ready to Use!

1. **Place your .tif images** in the input folders
2. **Run the OCR**: `python batch_ocr.py input output`
3. **Check results** in the output folder

## 🔮 Future GUI Integration

The current scripts are designed for easy GUI integration:
- Modular design with separate processor classes
- Progress callbacks available
- Configurable parameters
- Error handling with user-friendly messages

## 📞 Need Help?

- Run `python batch_ocr.py --help` for command options
- Check the detailed `README.md` for more information
- Test with `python demo_test.py` if you encounter issues

---

**Your CPU-optimized Qwen 2.5VL OCR setup is ready for production use! 🚀**
