"""
Memory-optimized Qwen 2.5VL Batch OCR Processor
Designed to handle large batch processing with limited memory by using sequential processing
and careful memory management.
"""

import os
import sys
import json
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import warnings
import gc

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*generation flags.*")

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
        logger.info(f"Loaded configuration from {config_path}")
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
                "max_new_tokens": 512,
                "do_sample": False,
                "max_workers": 2
            },
            "ocr_prompt": "Extract all text from this image accurately. Maintain formatting and structure."
        }

class MemoryOptimizedOCRProcessor:
    """Memory-optimized OCR processor that loads model once and processes sequentially"""
    
    def __init__(self, config: Dict = None):
        """Initialize the OCR processor"""
        if config is None:
            config = load_config()
        
        self.config = config
        self.model_name = config["model_settings"]["model_name"]
        self.device = self._setup_device(config["model_settings"]["device"])
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
            # Set fewer threads for memory optimization
            num_threads = min(1, multiprocessing.cpu_count())
            torch.set_num_threads(num_threads)
            logger.info(f"Set PyTorch to use {num_threads} CPU threads for memory optimization")
        
        return device
    
    def _load_model(self):
        """Load the Qwen 2.5VL model and components"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with memory optimization
            if self.device == "cpu":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Preprocess image for OCR with memory optimization"""
        try:
            # Load image
            if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
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
            
            # Resize large images to reduce memory usage
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(s * ratio) for s in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {image_path} to {new_size} for memory optimization")
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def extract_text(self, image: Image.Image, prompt: str = None) -> str:
        """Extract text from image with memory optimization"""
        try:
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
            
            # Get generation settings
            processing_settings = self.config.get("processing_settings", {})
            max_new_tokens = processing_settings.get("max_new_tokens", 512)
            do_sample = processing_settings.get("do_sample", False)
            
            # Generate text with memory optimization
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False  # Disable cache to save memory
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
            
            # Clear memory
            del inputs, output, generated_ids
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            # Clear memory on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            return ""
    
    def process_single_image(self, image_path: str, output_dir: str) -> Dict:
        """Process a single image and save results"""
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

class MemoryOptimizedBatchManager:
    """Memory-optimized batch OCR manager"""
    
    def __init__(self, input_dir: str, output_dir: str, config: Dict = None):
        """Initialize batch OCR manager"""
        if config is None:
            config = load_config()
        
        self.config = config
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR processor
        self.ocr_processor = MemoryOptimizedOCRProcessor(config=config)
    
    def scan_images(self) -> List[Tuple[str, str]]:
        """Scan input directory for images"""
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
        """Process all images sequentially for memory optimization"""
        image_files = self.scan_images()
        if not image_files:
            logger.warning("No images found to process")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        results = []
        failed_files = []
        
        logger.info("Starting sequential processing for memory optimization")
        
        # Process images one by one with progress bar
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for img_path, out_dir in image_files:
                result = self.ocr_processor.process_single_image(img_path, out_dir)
                results.append(result)
                
                if not result['success']:
                    failed_files.append(result['image_path'])
                
                pbar.update(1)
                
                # Force garbage collection after each image
                gc.collect()
        
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
    
    parser = argparse.ArgumentParser(description="Memory-Optimized Qwen 2.5VL Batch OCR Processor")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for OCR results")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        # Override config with command line arguments if provided
        if args.model:
            config["model_settings"]["model_name"] = args.model
        if args.device:
            config["model_settings"]["device"] = args.device
        
        # Create batch manager
        manager = MemoryOptimizedBatchManager(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
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
