# Automated Borescope Inspection for Pump Components Using Edge-AI

An edge-AI based computer vision system for automated inspection of internal pump component surfaces using borescope images. The system classifies component views as **OK** or **DEFECT** and combines three inspection angles using an all-views-required decision rule.

## Overview

Manual inspection of deep internal pump bores using flexible borescopes can be affected by operator fatigue, glare from metallic surfaces, changing illumination, and subjective judgement.

This project implements a lightweight computer vision pipeline that performs automated binary classification of pump component images. The trained model is converted to an optimized **ONNX** format for local edge inference, reducing dependency on cloud connectivity during inspection.

## System Workflow

```text
Borescope Images
       │
       ▼
Image Preprocessing
       │
       ▼
MobileNetV3-Small
       │
       ▼
OK / DEFECT Classification
       │
       ▼
Three-View Validation
       │
       ▼
   AND-Gate Logic
       │
   ┌───┴───┐
   ▼       ▼
  OK     DEFECT
```

Each pump component is evaluated from three different views. The component passes only when **all three views are classified as acceptable**. If any view contains a defect, the final component status becomes `DEFECT`.

## Key Features

* Automated binary classification of pump component images.
* Three-angle inspection to reduce camera blind spots.
* MobileNetV3-Small based image classification.
* ONNX model for lightweight local inference.
* Image preprocessing using RGB conversion and standardized `224 × 224` input dimensions.
* Pixel normalization and statistical scaling before inference.
* Unified visual output for the three inspected views.
* Automated final `OK` / `DEFECT` decision using AND-gate logic.

## Project Structure

```text
automated-borescope-inspection/
│
├── calculate_accuracy.py
├── verify_component.py
├── mobilenetv3_borescope.onnx
├── mobilenetv3_borescope.onnx.data
├── test_deck/
│   ├── DEFECT/
│   └── OK/
├── .gitignore
└── README.md
```

## Technologies

* **Python**
* **OpenCV** — image processing and preprocessing
* **NumPy** — image and array operations
* **MobileNetV3-Small** — lightweight image classification architecture
* **ONNX** — optimized model serialization and local inference

## Inspection Logic

The inspection system maintains a pass state for each of the three camera views.

```text
View 1 = 1
View 2 = 1
View 3 = 1

1 × 1 × 1 = 1 → OK
```

If any view detects a defect:

```text
View 1 = 1
View 2 = 0
View 3 = 1

1 × 0 × 1 = 0 → DEFECT
```

This ensures that a component is accepted only when every required inspection view passes.

## Image Preprocessing

Before inference, each input image is:

1. Converted from OpenCV's BGR format to RGB.
2. Resized to `224 × 224` pixels.
3. Converted to floating-point representation.
4. Normalized to the `[0, 1]` range.
5. Standardized using the required training mean and standard deviation.

This ensures that runtime inputs match the format expected by the trained model.

## Evaluation

The project includes a test deck containing:

* **10 defective component images**
* **10 acceptable component images**

The recorded proof-of-concept evaluation achieved:

| Metric                                 | Result |
| -------------------------------------- | -----: |
| Components evaluated                   |     20 |
| Defects correctly detected             |     10 |
| Acceptable components correctly passed |     10 |
| False alarms                           |      0 |
| Missed defects                         |      0 |
| Test deck accuracy                     |   100% |

The project report also records a **projected long-term fleet accuracy of 94.25%**, accounting for production factors such as camera lens contamination, fixture variation, and illumination changes.

## Running the Inspection

Install the required Python dependencies for the scripts, then run the component verification script with three image paths:

```bash
python verify_component.py <image1> <image2> <image3>
```

The program processes the three views independently and produces a combined inspection result.

To evaluate the test deck, use:

```bash
python calculate_accuracy.py
```

> The exact command-line arguments may depend on the implementation in the current scripts.

## Edge Deployment

The trained model is serialized into ONNX format so that the production-side runtime can perform local inference without requiring continuous cloud communication.

The project targets low-latency inference suitable for factory-floor deployment. The implementation report records inference execution of under **45 ms per image** on the tested local computer architecture.

## Future Improvements

* Integrate three physical borescopes directly into the inspection fixture.
* Connect the inspection result to a manufacturing PLC using Modbus or TCP/IP.
* Trigger an automated reject mechanism for defective components.
* Collect production false-alarm images for continuous model improvement.
* Improve long-term production precision through controlled retraining.

## Project Context

Developed as an internship project at **Delphi-TVS Technologies, Oragadam**.

---
