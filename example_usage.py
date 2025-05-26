"""
Simple example script for testing Qwen 2.5VL OCR
"""

import os
from batch_ocr import BatchOCRManager

def create_test_structure():
    """Create test directory structure as described"""
    test_input = "input"
    
    # Create the folder structure
    folders = [
        "input/folder1/subfolder1",
        "input/folder1/subfolder2", 
        "input/folder2"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")
    
    print("\nTest structure created!")
    print("Please place your .tif images in the appropriate folders:")
    print("- input/folder1/subfolder1/")
    print("- input/folder1/subfolder2/")
    print("- input/folder2/")

def run_ocr_example():
    """Run OCR on the test structure"""
    input_dir = "input"
    output_dir = "output"
    
    if not os.path.exists(input_dir):
        print("Input directory not found. Creating test structure...")
        create_test_structure()
        return
    
    print("Starting OCR processing...")
    
    # Initialize and run batch OCR
    manager = BatchOCRManager(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=4  # Adjust based on your CPU
    )
    
    summary = manager.process_batch()
    
    print("\n" + "="*50)
    print("OCR PROCESSING COMPLETED")
    print("="*50)
    print(f"Total files processed: {summary['total']}")
    print(f"Successful: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Check if input folder exists
    if not os.path.exists("input"):
        choice = input("Input folder not found. Create test structure? (y/n): ")
        if choice.lower() == 'y':
            create_test_structure()
        else:
            print("Please create input folder with your images and run again.")
    else:
        run_ocr_example()
