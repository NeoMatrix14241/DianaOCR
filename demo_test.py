"""
Demo script to test Qwen 2.5VL model loading and basic functionality
This script tests the OCR setup without requiring actual images
"""

import sys
import time
import logging
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        # Try to load a larger font
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Add some sample text
    text_lines = [
        "SAMPLE DOCUMENT",
        "Invoice #12345",
        "Date: May 26, 2025",
        "Customer: John Doe",
        "Amount: $1,234.56",
        "",
        "This is a test image for OCR processing.",
        "The Qwen 2.5VL model should be able to",
        "extract this text accurately."
    ]
    
    y_position = 50
    for line in text_lines:
        if line.strip():  # Skip empty lines
            draw.text((50, y_position), line, fill='black', font=font)
        y_position += 50
    
    return img

def test_model_loading():
    """Test loading the Qwen 2.5VL model"""
    logger.info("Testing Qwen 2.5VL model loading...")
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        
        # Use a smaller model for testing to reduce download time and memory usage
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        logger.info(f"Loading model: {model_name}")
        
        # Load processor and tokenizer
        logger.info("Loading processor and tokenizer...")
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Processor and tokenizer loaded successfully")
        
        # Load model
        logger.info("Loading model (this may take a while on first run)...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None
        )
        model = model.to("cpu")
        model.eval()
        logger.info("✓ Model loaded successfully")
        
        return model, processor, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None

def test_ocr_functionality(model, processor, tokenizer):
    """Test OCR functionality with a sample image"""
    if model is None:
        logger.error("Model not loaded, skipping OCR test")
        return False
    
    try:
        logger.info("Creating test image...")
        test_image = create_test_image()
        
        # Save test image for reference
        test_image.save("test_image.png")
        logger.info("✓ Test image saved as test_image.png")
        
        logger.info("Testing OCR extraction...")
        
        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "Extract all text from this image accurately. Maintain formatting and structure."}
                ]
            }
        ]
        
        # Process inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[test_image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cpu")
        
        # Generate text
        logger.info("Generating OCR output...")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
        
        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output)
        ]
        response = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        logger.info("✓ OCR extraction completed")
        logger.info(f"Extracted text:\n{'-'*40}\n{response}\n{'-'*40}")
        
        # Save results
        with open("test_ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(response)
        logger.info("✓ OCR result saved to test_ocr_result.txt")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during OCR test: {str(e)}")
        return False

def main():
    """Main demo function"""
    print("="*60)
    print("Qwen 2.5VL OCR Demo - Model Loading Test")
    print("="*60)
    
    # Check system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: CPU (using {torch.get_num_threads()} threads)")
    
    start_time = time.time()
    
    # Test model loading
    model, processor, tokenizer = test_model_loading()
    
    if model is not None:
        logger.info(f"Model loading time: {time.time() - start_time:.2f} seconds")
        
        # Test OCR functionality
        ocr_start = time.time()
        ocr_success = test_ocr_functionality(model, processor, tokenizer)
        
        if ocr_success:
            logger.info(f"OCR test time: {time.time() - ocr_start:.2f} seconds")
            logger.info(f"Total demo time: {time.time() - start_time:.2f} seconds")
            
            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Your Qwen 2.5VL OCR setup is working correctly.")
            print("You can now place your .tif images in the input folders and run:")
            print("  python batch_ocr.py input output")
            print("\nDemo files created:")
            print("  - test_image.png (sample test image)")
            print("  - test_ocr_result.txt (OCR output)")
        else:
            print("\n❌ OCR test failed")
    else:
        print("\n❌ Model loading failed")
        print("Please check your internet connection and try again.")
        print("The model will be downloaded from Hugging Face on first run.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Demo failed with error: {e}")
