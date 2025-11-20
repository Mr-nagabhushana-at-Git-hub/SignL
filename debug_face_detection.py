#!/usr/bin/env python3
# debug_face_detection.py - Test face recognition with your directory structure

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import logging

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from majorSignL.models.face_processor import FaceProcessor
    print("✅ Successfully imported FaceProcessor")
except ImportError as e:
    print(f"❌ Failed to import FaceProcessor: {e}")
    print("Make sure you're in the project root and have activated the SignL environment")
    sys.exit(1)

def main():
    print("🧪 MajorSignL Face Recognition Debug Tool")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get project paths
    project_root = Path(__file__).parent
    data_path = project_root / "src" / "data" / "models"
    fase_data_path = project_root / "src" / "data" / "fase_data"
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Data path: {data_path}")
    print(f"📁 Face data path: {fase_data_path}")
    print()
    
    # Check if fase_data exists and list person folders
    if not fase_data_path.exists():
        print(f"❌ Face data directory not found: {fase_data_path}")
        return
    
    person_folders = [f for f in fase_data_path.iterdir() 
                     if f.is_dir() and not f.name.endswith('.pkl')]
    
    print(f"👥 Found {len(person_folders)} person folders:")
    for folder in person_folders:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(folder.glob(ext))
        print(f"  - {folder.name}: {len(image_files)} images")
    print()
    
    # Initialize FaceProcessor
    print("🔄 Initializing FaceProcessor...")
    try:
        face_processor = FaceProcessor(data_path)
        face_info = face_processor.get_known_faces_info()
        
        print(f"✅ FaceProcessor initialized successfully!")
        print(f"📊 Total faces loaded: {face_info['total_faces']}")
        print(f"👥 Known faces: {', '.join(face_info['names'])}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize FaceProcessor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with a sample image
    print("🖼️ Testing with sample images...")
    test_count = 0
    success_count = 0
    
    for person_folder in person_folders[:3]:  # Test first 3 people
        image_files = list(person_folder.glob('*.jpg'))[:2]  # Test first 2 images per person
        
        for img_file in image_files:
            test_count += 1
            print(f"Testing: {img_file.name} (expected: {person_folder.name})")
            
            try:
                # Load and test image
                frame = cv2.imread(str(img_file))
                if frame is None:
                    print(f"  ❌ Failed to load image")
                    continue
                
                # Process frame
                face_data = face_processor.process_frame(frame)
                
                if face_data:
                    for face in face_data:
                        name = face['name']
                        confidence = face['confidence']
                        box = face['box']
                        
                        print(f"  ✅ Detected: {name} (confidence: {confidence:.3f})")
                        
                        if name == person_folder.name:
                            success_count += 1
                            print(f"  🎯 Correct match!")
                        elif name != "Unknown":
                            print(f"  ❌ Incorrect match (expected {person_folder.name})")
                        else:
                            print(f"  ❓ Unknown person")
                        
                        # Draw bounding box and save result
                        left, top, right, bottom = box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", 
                                  (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save result
                        output_path = project_root / f"debug_result_{img_file.stem}.jpg"
                        cv2.imwrite(str(output_path), frame)
                        print(f"  💾 Saved result to: {output_path}")
                else:
                    print(f"  ❌ No faces detected")
                    
            except Exception as e:
                print(f"  ❌ Error processing {img_file.name}: {e}")
            
            print()
    
    # Summary
    print("📈 Test Summary:")
    print(f"  Total tests: {test_count}")
    print(f"  Successful recognitions: {success_count}")
    print(f"  Success rate: {(success_count/test_count)*100:.1f}%" if test_count > 0 else "  No tests run")
    
    print("\n🎯 Debug complete! Check the debug_result_*.jpg files for visual results.")

if __name__ == "__main__":
    main()
