import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "mobilenetv3_borescope.onnx"
TEST_DECK_PATH = "./test_deck"

if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found!")
    exit()

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

tp, tn, fp, fn = 0, 0, 0, 0

# STRICT MAPPING: PyTorch alphabetization forces DEFECT=0, OK=1
class_mapping = {"DEFECT": 0, "OK": 1}

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

for folder_name in ["DEFECT", "OK"]:
    folder_path = os.path.join(TEST_DECK_PATH, folder_name)
    if not os.path.exists(folder_path):
        continue
        
    actual_class = class_mapping[folder_name]
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (224, 224))
            tensor = resized.astype(np.float32) / 255.0
            tensor = (tensor - mean) / std
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)
            
            outputs = session.run(['output'], {'input': tensor})
            predicted_class = np.argmax(outputs[0][0])
            
            if actual_class == 0:  # Ground Truth: DEFECT
                if predicted_class == 0: tp += 1
                else: fn += 1  # AI guessed 1 (OK) -> Missed Defect!
            else:                  # Ground Truth: OK
                if predicted_class == 1: tn += 1
                else: fp += 1  # AI guessed 0 (DEFECT) -> False Alarm!

total_images = tp + tn + fp + fn
accuracy = ((tp + tn) / total_images) * 100 if total_images > 0 else 0

print("=" * 50)
print(f"   📊 DELPHI-TVS CORRECTED QA REPORT")
print("=" * 50)
print(f"Total Components Inspected : {total_images}")
print(f"Correctly Caught Defects   : {tp}")
print(f"Correctly Passed Clean     : {tn}")
print("-" * 50)
print(f"CRITICAL: Missed Defects (Escapees) : {fn}")
print(f"False Alarms (Clean flagged bad)    : {fp}")
print("-" * 50)
print(f"Real Test Deck Accuracy             : {accuracy:.2f}%")
print("=" * 50)