"""
Ultra Memory-Optimized Qwen 2.5VL Batch OCR Processor
Extremely aggressive memory optimization for limited RAM environments
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional
import logging
import warnings

# Aggressive memory optimization settings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import cv2
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing_minimal.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UltraMinimalOCRProcessor:
    """Ultra memory-optimized OCR processor"""
    
    def __init__(self):
        """Initialize with aggressive memory optimization"""
        # Force single thread and minimal memory usage
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # Use CPU only with minimal precision
        self.device = "cpu"
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        
        logger.info("Initializing ultra-minimal OCR processor")
        self._load_model()
    
    def _load_model(self):
        """Load model with maximum memory optimization"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load components individually to manage memory
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=False
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False
            )
            
            # Load model with aggressive memory settings
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None,
                low_cpu_mem_usage=True,
                local_files_only=False,
                trust_remote_code=True
            )
            
            # Move to CPU and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Model loaded successfully with ultra-minimal settings")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Preprocess image with aggressive size reduction"""
        try:
            # Load image
            if image_path.lower().endswith(('.tif', '.tiff')):
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
            
            # Aggressive size reduction for memory
            max_size = 1024  # Smaller than before
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(s * ratio) for s in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {new_size} for ultra-minimal memory usage")
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def extract_text_minimal(self, image: Image.Image) -> str:
        """Extract text with minimal memory footprint"""
        try:
            # Simple prompt to minimize token usage
            prompt = "Extract text from this image."
            
            # Prepare minimal conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with minimal settings
            text_prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate with very conservative settings
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced from 512
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,  # Disable cache
                    output_attentions=False,  # Disable attention outputs
                    output_hidden_states=False,  # Disable hidden state outputs
                    return_dict_in_generate=False  # Minimal return format
                )
            
            # Decode minimal output
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output)
            ]
            
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            
            # Aggressive cleanup
            del inputs, output, generated_ids
            gc.collect()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            # Cleanup on error
            gc.collect()
            return ""
    
    def process_single_image(self, image_path: str, output_dir: str) -> Dict:
        """Process single image with minimal memory usage"""
        start_time = time.time()
        result = {
            'image_path': image_path,
            'success': False,
            'text': '',
            'processing_time': 0,
            'error': None
        }
        
        try:
            logger.info(f"Processing: {image_path}")
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                result['error'] = "Failed to preprocess image"
                return result
            
            # Extract text with minimal settings
            extracted_text = self.extract_text_minimal(image)
            
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
            
            logger.info(f"SUCCESS: {image_path} -> {len(extracted_text)} chars extracted")
            
            # Cleanup after each image
            del image
            gc.collect()
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            logger.error(f"ERROR processing {image_path}: {str(e)}")
            gc.collect()
        
        return result

def scan_images(input_dir: str) -> List[tuple]:
    """Scan for images to process"""
    input_path = Path(input_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_path = Path(root) / file
                rel_path = image_path.relative_to(input_path)
                output_subdir = Path("output") / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                image_files.append((str(image_path), str(output_subdir)))
    
    return image_files

def main():
    """Main processing function"""
    try:
        # Setup
        input_dir = "input"
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Starting ultra-minimal OCR processing")
        
        # Initialize processor
        processor = UltraMinimalOCRProcessor()
        
        # Find images
        image_files = scan_images(input_dir)
        logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.warning("No images found!")
            return
        
        # Process images sequentially with progress tracking
        results = []
        successful = 0
        failed = 0
        
        with tqdm(total=len(image_files), desc="OCR Processing") as pbar:
            for i, (img_path, out_dir) in enumerate(image_files):
                logger.info(f"Progress: {i+1}/{len(image_files)}")
                
                result = processor.process_single_image(img_path, out_dir)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    logger.info(f"✓ SUCCESS: {result['text'][:100]}...")
                else:
                    failed += 1
                    logger.error(f"✗ FAILED: {result['error']}")
                
                pbar.update(1)
                
                # Force memory cleanup every few images
                if (i + 1) % 5 == 0:
                    gc.collect()
                    logger.info(f"Memory cleanup at image {i+1}")
        
        # Summary
        logger.info(f"COMPLETED: {successful}/{len(image_files)} successful, {failed} failed")
        
        # Save results
        summary = {
            'total': len(image_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
        with open(output_dir / 'ultra_minimal_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ULTRA-MINIMAL OCR PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {output_dir / 'ultra_minimal_results.json'}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
