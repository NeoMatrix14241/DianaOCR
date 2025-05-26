"""
Qwen 2.5VL Batch OCR Processor
A simple script for batch OCR processing using Qwen 2.5VL model on CPU with multi-threading support.
Supports folder structure analysis and automatic GPU detection.
"""

import os
import sys
import json
import time
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import logging

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"✓ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return {
            "model_settings": {
                "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                "device": "auto"
            },
            "processing_settings": {
                "max_new_tokens": 1024,
                "temperature": 0.1,
                "do_sample": False
            },
            "ocr_prompt": "Extract all text from this image accurately. Maintain formatting and structure."
        }

class QwenVLOCRProcessor:
    """Main OCR processor class using Qwen 2.5VL model"""
    
    def __init__(self, model_name: str = None, device: str = None, config: Dict = None):
        """
        Initialize the OCR processor
        
        Args:
            model_name: Hugging Face model name for Qwen 2.5VL (overrides config)
            device: Device to use ('auto', 'cpu', 'cuda') (overrides config)
            config: Configuration dictionary
        """
        # Load config if not provided
        if config is None:
            config = load_config()
        
        self.config = config
        self.model_name = model_name or config["model_settings"]["model_name"]
        self.device = self._setup_device(device or config["model_settings"]["device"])
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        logger.info(f"Using model: {self.model_name}")
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup and return the appropriate device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU")
        else:
            logger.info(f"Using specified device: {device}")
        
        if device == "cpu":
            # Set number of threads for CPU optimization
            num_threads = multiprocessing.cpu_count()
            torch.set_num_threads(num_threads)
            logger.info(f"Set PyTorch to use {num_threads} CPU threads")
        
        return device
    
    def _load_model(self):
        """Load the Qwen 2.5VL model and components"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate settings
            if self.device == "cpu":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map=None
                )
                self.model = self.model.to(self.device)
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Preprocess image for OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object or None if error
        """
        try:
            # Load image
            if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
                # Handle TIFF files with OpenCV for better support
                img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img_array is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_array)
            else:
                image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def extract_text(self, image: Image.Image, prompt: str = None) -> str:
        """
        Extract text from image using Qwen 2.5VL
        
        Args:
            image: PIL Image object
            prompt: OCR prompt for the model (uses config default if None)
            
        Returns:
            Extracted text as string
        """
        try:
            # Use prompt from config if not provided
            if prompt is None:
                prompt = self.config.get("ocr_prompt", "Extract all text from this image accurately. Maintain formatting and structure.")
            
            # Prepare conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Get generation settings from config
            processing_settings = self.config.get("processing_settings", {})
            max_new_tokens = processing_settings.get("max_new_tokens", 1024)
            do_sample = processing_settings.get("do_sample", False)
            temperature = processing_settings.get("temperature", 0.1)
            
            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature
                )
            
            # Decode output
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output)
            ]
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def process_single_image(self, image_path: str, output_dir: str) -> Dict:
        """
        Process a single image and save results
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save results
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        result = {
            'image_path': image_path,
            'success': False,
            'text': '',
            'processing_time': 0,
            'error': None
        }
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                result['error'] = "Failed to preprocess image"
                return result
            
            # Extract text
            extracted_text = self.extract_text(image)
            
            # Save results
            image_name = Path(image_path).stem
            txt_path = Path(output_dir) / f"{image_name}.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            result.update({
                'success': True,
                'text': extracted_text,
                'output_file': str(txt_path),
                'processing_time': time.time() - start_time
            })
            
            logger.info(f"Processed: {image_path} -> {txt_path}")
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            logger.error(f"Error processing {image_path}: {str(e)}")
        
        return result

class BatchOCRManager:
    """Manager for batch OCR processing"""
    
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = None, config: Dict = None):
        """
        Initialize batch OCR manager
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            max_workers: Maximum number of worker threads
            config: Configuration dictionary
        """
        # Load config if not provided
        if config is None:
            config = load_config()
        
        self.config = config
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Get max_workers from config if not specified
        if max_workers is None:
            max_workers = config.get("processing_settings", {}).get("max_workers")
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR processor with config
        self.ocr_processor = QwenVLOCRProcessor(config=config)
    
    def scan_images(self) -> List[Tuple[str, str]]:
        """
        Scan input directory for images and determine folder structure
        
        Returns:
            List of tuples (image_path, relative_output_path)
        """
        # Get supported formats from config
        supported_formats = self.config.get("supported_formats", [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"])
        image_extensions = set(fmt.lower() for fmt in supported_formats)
        image_files = []
        
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_path = Path(root) / file
                    
                    # Calculate relative path for output structure
                    rel_path = image_path.relative_to(self.input_dir)
                    output_subdir = self.output_dir / rel_path.parent
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    
                    image_files.append((str(image_path), str(output_subdir)))
        
        logger.info(f"Found {len(image_files)} images to process")
        return image_files
    
    def process_batch(self) -> Dict:
        """
        Process all images in batch
        
        Returns:
            Summary statistics
        """
        image_files = self.scan_images()
        if not image_files:
            logger.warning("No images found to process")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        results = []
        failed_files = []
        
        logger.info(f"Starting batch processing with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.ocr_processor.process_single_image, img_path, out_dir): img_path
                for img_path, out_dir in image_files
            }
            
            # Process results with progress bar
            with tqdm(total=len(image_files), desc="Processing images") as pbar:
                for future in as_completed(future_to_image):
                    result = future.result()
                    results.append(result)
                    
                    if not result['success']:
                        failed_files.append(result['image_path'])
                    
                    pbar.update(1)
        
        # Generate summary
        total_files = len(results)
        successful_files = sum(1 for r in results if r['success'])
        failed_count = total_files - successful_files
        
        summary = {
            'total': total_files,
            'success': successful_files,
            'failed': failed_count,
            'processing_time': sum(r['processing_time'] for r in results),
            'failed_files': failed_files
        }
        
        # Save detailed results
        results_file = self.output_dir / 'processing_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed: {successful_files}/{total_files} successful")
        if failed_files:
            logger.warning(f"Failed files: {failed_files}")
        
        return summary

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen 2.5VL Batch OCR Processor")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for OCR results")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct", help="Model name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        # Override model if specified
        if args.model != "Qwen/Qwen2-VL-7B-Instruct":  # If user specified a different model
            config["model_settings"]["model_name"] = args.model
        
        # Override device if specified  
        if args.device != "auto":
            config["model_settings"]["device"] = args.device
        
        # Create batch manager with config
        manager = BatchOCRManager(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_workers=args.workers,
            config=config
        )
        
        # Process batch
        summary = manager.process_batch()
        
        print("\n" + "="*50)
        print("BATCH PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files: {summary['total']}")
        print(f"Successful: {summary['success']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total processing time: {summary['processing_time']:.2f} seconds")
        print(f"Results saved to: {manager.output_dir}")
        
        if summary['failed'] > 0:
            print(f"\nFailed files saved to: {manager.output_dir}/processing_results.json")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
