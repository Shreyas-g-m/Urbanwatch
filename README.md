# 🛰️ UrbanWatch

### AI-Powered Multi-Task Remote Sensing Analysis

UrbanWatch is an end-to-end remote sensing AI pipeline designed to analyze satellite imagery for **building detection, land-cover semantic segmentation, and temporal change detection**.

The project combines three complementary computer vision tasks:

- 🏢 **SpaceNet 2** — Building Detection using YOLO
- 🌍 **LoveDA** — Land-Cover Semantic Segmentation using DeepLabV3 + ResNet18
- 🔄 **LEVIR-CD+** — Bi-temporal Change Detection using a Siamese deep learning model
- 🤖 **Llama 3.2** — AI assistant for interpreting UrbanWatch results
- 🖥️ **Streamlit** — Interactive application interface

---

## 🚀 Project Overview

UrbanWatch processes satellite imagery through multiple specialized AI models and provides a unified framework for understanding urban environments.

```text
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
