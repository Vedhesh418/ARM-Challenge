import sys
import os
import cv2
import numpy as np
import onnxruntime as ort

# 1. Require exactly 3 images as arguments
if len(sys.argv) < 4:
    print("\n❌ Error: Missing arguments!")
    print("Usage: python3 verify_component.py <image_angle1> <image_angle2> <image_angle3>")
    exit()

image_paths = [sys.argv[1], sys.argv[2], sys.argv[3]]
MODEL_PATH = "mobilenetv3_borescope.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model engine '{MODEL_PATH}' not found.")
    exit()

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# Strict indexing: 0 = DEFECT, 1 = OK
labels = ["DEFECT", "OK"]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Global component state (True means acceptable so far)
component_is_ok = True
individual_results = []
display_images = []

print("\n" + "="*50)
print("COMPONENT CHECK")
print("="*50)

# 2. Process all 3 images sequentially
for i, path in enumerate(image_paths):
    if not os.path.exists(path):
        print(f"❌ Error: Image for Angle {i+1} not found at '{path}'")
        exit()
        
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Error: Cannot read image at '{path}'")
        exit()
        
    # Run model pre-processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    
    # AI Inference
    outputs = session.run(['output'], {'input': tensor})
    predicted_index = np.argmax(outputs[0][0])
    decision = labels[predicted_index]
    
    individual_results.append(decision)
    print(f"Image {i+1} [{os.path.basename(path)}]: {decision}")
    
    # STRICT RULE: If even one angle fails, the component fails
    if decision == "DEFECT":
        component_is_ok = False

    # Prep the individual image window for display later
    # Draw a small text tag directly on each sub-view image
    tag_color = (0, 255, 0) if decision == "OK" else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (180, 40), (30, 30, 30), -1)
    cv2.putText(img, f"View {i+1}: {decision}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2)
    
    # Resize to a consistent height for the side-by-side display monitor
    display_images.append(cv2.resize(img, (300, 300)))

# 3. Final Multi-View Aggregation Verdict
final_status = "PASS" if component_is_ok else "DEFECTIVE"
final_color = (0, 255, 0) if component_is_ok else (0, 0, 255)

print("-" * 50)
print(f"FINAL COMPONENT VERDICT: {final_status}")
print("=" * 50 + "\n")

# --- MULTI-PANEL DISPLAY FOR THE MANAGER ---
# Combine all three 300x300 images horizontally into a single layout strip
horizontal_strip = np.hstack(display_images)

# Create a master status bar to paste at the top
header_bar = np.zeros((70, horizontal_strip.shape[1], 3), dtype=np.uint8) + 30
header_text = f"INSPECTION RESULT: {final_status}"
cv2.putText(header_bar, header_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2, cv2.LINE_AA)

# Stack the status bar on top of our image strip
final_dashboard = np.vstack([header_bar, horizontal_strip])

# Pop up the unified cockpit screen
cv2.imshow("Delphi-TVS Multi-Angle QA Monitor", final_dashboard)
print("👉 Click on the dashboard window and press ANY KEY to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()