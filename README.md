UrbanWatch

AI-Powered Multi-Task Remote Sensing Analysis

UrbanWatch is an end-to-end remote sensing AI pipeline designed to analyze satellite imagery for building detection, land-cover semantic segmentation, and temporal change detection.

The project combines three complementary computer vision tasks:

🏢 SpaceNet 2 — Building Detection using YOLO

🌍 LoveDA — Land-Cover Semantic Segmentation using DeepLabV3 + ResNet18

🔄 LEVIR-CD+ — Bi-temporal Change Detection using a Siamese deep learning model

🤖 Llama 3.2 — AI assistant for interpreting UrbanWatch results

🖥️ Streamlit — Interactive application interface

🚀 Project Overview

UrbanWatch processes satellite imagery through multiple specialized AI models and provides a unified framework for understanding urban environments.

                    ┌──────────────────────┐
                    │   Satellite Imagery  │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │ SpaceNet 2  │   │   LoveDA    │   │ LEVIR-CD+   │
      │    YOLO     │   │ DeepLabV3   │   │   Siamese   │
      │ Buildings   │   │ Land Cover  │   │   Changes   │
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ UrbanWatch Analysis  │
                    └──────────┬───────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
              Streamlit App        Llama AI Analyst
              # 🛰️ UrbanWatch — Multi-Task Urban Intelligence System

UrbanWatch is an end-to-end **remote sensing and computer vision platform** that combines building detection, land-cover semantic segmentation, and satellite image change detection into a unified system.

It also includes a **local Llama 3.2 AI analyst** and an interactive **Streamlit application** for model inference, evaluation, visualization, and analysis.

---

## 📊 Final Results

### 🏢 SpaceNet 2 — Building Detection

**Model:** YOLOv8m

#### Dataset

| Metric                      |  Value |
| --------------------------- | -----: |
| Total images                |  1,148 |
| Images with building labels |    633 |
| Training images             |    919 |
| Validation images           |    229 |
| Training building boxes     | 12,980 |
| Validation building boxes   |  3,402 |

#### Results

| Metric    |      Score |
| --------- | ---------: |
| Precision | **0.8274** |
| Recall    | **0.7516** |
| mAP@50    | **0.8212** |
| mAP@50-95 | **0.5004** |

The building detector was trained using **SpaceNet 2 Paris imagery**, with building annotations converted into YOLO-compatible bounding boxes.

---

### 🌍 LoveDA — Semantic Segmentation

**Model:** DeepLabV3 + ResNet18

LoveDA provides urban and rural land-cover imagery for semantic segmentation.

#### Dataset

| Metric                   |  Value |
| ------------------------ | -----: |
| Training source images   |  2,522 |
| Validation source images |  1,669 |
| Training 512×512 crops   | 10,088 |
| Validation 512×512 crops |  6,676 |
| Number of classes        |      8 |

#### Results

| Metric         |      Score |
| -------------- | ---------: |
| mIoU           | **0.5402** |
| Mean Dice/F1   | **0.6847** |
| Pixel Accuracy | **0.6880** |

The segmentation pipeline uses **512×512 crops** to make training feasible on an 8 GB GPU.

---

### 🔄 LEVIR-CD+ — Change Detection

**Model:** Siamese Deep Learning Change-Detection Network

LEVIR-CD+ provides pairs of satellite images captured at different points in time. The model learns to identify pixels representing meaningful changes between the images.

#### Dataset

| Metric             |     Value |
| ------------------ | --------: |
| Training pairs     |       637 |
| Training split     |       510 |
| Validation split   |       127 |
| Test pairs         |       348 |
| Image size         | 1024×1024 |
| Image channels     |       RGB |
| Change-mask values |   0 / 255 |

#### Results

| Metric         |      Score |
| -------------- | ---------: |
| Precision      | **0.7070** |
| Recall         | **0.7310** |
| Dice/F1        | **0.7188** |
| IoU            | **0.5610** |
| Pixel Accuracy | **0.9764** |

The final model was evaluated on all **348 held-out test pairs**.

---

### 🎯 SAM 2 — Promptable Segmentation

**Model:** Segment Anything Model 2 (SAM 2)

SAM 2 extends UrbanWatch with promptable, high-quality segmentation. It can be used to refine and inspect objects or regions identified in satellite imagery, providing a more flexible segmentation workflow alongside the project's task-specific trained models.

The SAM 2 integration complements the existing SpaceNet 2, LoveDA, and LEVIR-CD+ pipelines rather than replacing them:

- **SpaceNet 2 / YOLOv8m** — trained building detection
- **LoveDA / DeepLabV3-ResNet18** — trained land-cover semantic segmentation
- **LEVIR-CD+ / Siamese network** — trained bi-temporal change detection
- **SAM 2** — promptable segmentation and object-level refinement

SAM 2 is integrated into the UrbanWatch inference workflow for interactive segmentation of selected satellite imagery.

---

## 🤖 Llama AI Analyst

UrbanWatch includes a local AI assistant powered by **Llama 3.2 through Ollama**.

The assistant is designed to answer questions about:

* Model performance
* Dataset statistics
* Evaluation metrics
* Differences between the three models
* UrbanWatch pipeline architecture
* Interpretation of results

### Example

```text
User:
Which UrbanWatch model performed best?

UrbanWatch AI:
The SpaceNet 2 building detector achieved the highest
reported mAP@50 at 0.8212, while LEVIR-CD+ achieved
an IoU of 0.5610 and LoveDA achieved an mIoU of 0.5402.
These metrics measure different tasks and should not be
treated as directly interchangeable.


The Llama model runs locally through Ollama, so the AI assistant does not require a cloud API key.

🖥️ Interactive Application

UrbanWatch includes a Streamlit application providing a unified interface for the project.

The application brings together:

🏢 Building detection

🌍 Land-cover segmentation

🔄 Change detection

🎯 SAM 2 promptable segmentation

📊 Model results

🖼️ Visual inference outputs

🤖 AI-assisted analysis

🧠 Technologies

Deep Learning

PyTorch

Torchvision

Ultralytics YOLO

DeepLabV3

ResNet18

SAM 2

Computer Vision

OpenCV

Pillow

NumPy

Geospatial / Remote Sensing

GeoJSON

TIFF imagery

Satellite image preprocessing

Raster-based semantic segmentation

AI Assistant

Llama 3.2

Ollama

Application

Streamlit

Development

Python

Git

GitHub

VS Code

📁 Project Structure

UrbanWatch/
│
├── 01_dataset_structure_analysis.ipynb
├── 02_dataset_sanity_check.py
│
├── 03_spacenet_inspection.py
├── 04_preprocess_spacenet.py
├── 05_verify_yolo.py
├── 06_train_yolo.py
│
├── 07_model_evaluation.ipynb
├── 08_yolo_evaluation.py
│
├── 09_levir_preprocessing.ipynb
├── 10_change_detection.py
├── 11_evaluate_change_detection.py
│
├── 12_loveda_preprocessing.ipynb
├── 13_preprocess_loveda.py
├── 14_train_loveda.py
├── 15_evaluate_loveda.py
│
├── 16_urbanwatch_inference.py
├── 17_change_detection_inference.py
├── 18_urbanwatch_final_report.py
├── 19_urbanwatch_app.py
├── 20_urbanwatch_llama.py
│
├── test_environment.py
├── requirements.txt
├── .gitignore
└── README.md


Large datasets, trained model weights, and generated results are intentionally excluded from the Git repository.

⚙️ Installation

1. Clone the repository

git clone https://github.com/Shreyas-g-m/UrbanWatch.git
cd UrbanWatch

2. Create the Conda environment

conda create -n urbanwatch python=3.10
conda activate urbanwatch

3. Install dependencies

pip install -r requirements.txt

🤖 Llama Setup

Install Ollama from:

https://ollama.com/

Then download Llama 3.2:

ollama pull llama3.2

Test the installation:

ollama run llama3.2

The Python integration uses the Ollama Python package:

pip install ollama

Run the UrbanWatch AI assistant:

python 20_urbanwatch_llama.py

🖥️ Running the Application

Start the Streamlit application:

streamlit run 19_urbanwatch_app.py

The application provides a unified interface for the UrbanWatch models.

📈 Evaluation Summary

DatasetTaskModelPrimary MetricResult









SpaceNet 2

Building Detection

YOLOv8m

mAP@50

0.8212

LoveDA

Semantic Segmentation

DeepLabV3-ResNet18

mIoU

0.5402

LEVIR-CD+

Change Detection

Siamese Network

IoU

0.5610

Note: These primary metrics belong to different computer vision tasks and therefore should not be interpreted as a direct ranking of the three models.

⚠️ Limitations

The three datasets represent different remote sensing tasks and geographic distributions.

Model metrics are task-specific and are not directly comparable across datasets.

LoveDA segmentation performance varies substantially between classes.

Pixel accuracy can be inflated when background pixels dominate.

The current unified inference demonstration processes selected imagery rather than representing a production-scale deployment.

SAM 2 provides promptable segmentation but does not replace the task-specific supervised models used for benchmark evaluation.

The models are research prototypes and should not be treated as operational geospatial decision systems without additional validation.

🔬 Future Work

Potential improvements include:

Stronger segmentation architectures

Higher-resolution inference

Improved small-object detection

Multi-scale satellite image processing

Better change-detection architectures

Cross-dataset domain adaptation

GIS integration

Interactive map visualization

Larger-scale inference

Cloud deployment

Improved Llama-based geospatial reasoning

Automated report generation

👨‍💻 Author

Shreyas Gouda M

BMS College of Engineering Computer Science and Engineering

GitHub: https://github.com/Shreyas-g-m

📜 Project Status

UrbanWatch — Core Pipeline Complete

✅ SpaceNet 2 preprocessing

✅ YOLO building detection

✅ YOLO evaluation

✅ LEVIR-CD+ preprocessing

✅ Change detection training

✅ LEVIR-CD+ test evaluation

✅ LoveDA preprocessing

✅ Semantic segmentation training

✅ LoveDA evaluation

✅ Unified inference

✅ SAM 2 integration

✅ Final evaluation report

✅ Streamlit application

✅ Local Llama AI assistant

✅ GitHub repository preparation

📌 Disclaimer

UrbanWatch is a research and educational computer vision project demonstrating multiple remote sensing workflows. The reported results are specific to the datasets, preprocessing pipelines, training configurations, and evaluation procedures used in this project.

The system should not be considered a production-ready geospatial decision-making system without additional validation on real-world data.
