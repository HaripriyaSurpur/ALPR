# NEXUS ANPR — Automatic Number Plate Recognition System

> AI-powered vehicle surveillance and threat detection platform  
> Built with YOLOv8 · EasyOCR · OpenCV · Flask · SQLite

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-purple)
![Flask](https://img.shields.io/badge/Backend-Flask-green)

## What it does
Detects and reads vehicle number plates from uploaded images,
classifies them as authorized / threat / unknown, and displays
results on a real-time surveillance dashboard.

## Tech stack
| Layer | Tools |
|-------|-------|
| Detection | YOLOv8, EAST, OpenCV contour fallback |
| OCR | EasyOCR |
| Backend | Flask, SQLite |
| Frontend | HTML, CSS, JavaScript |

## Features
- Multi-stage plate detection pipeline (YOLOv8 → EAST → contour)
- Blacklist-based threat classification with alert system
- VIP / official vehicle tagging
- Indian state origin mapping (all 36 states/UTs)
- Historical scan analytics dashboard
- 81% authorized · 19% threats across 26 test scans

## Setup
ultralytics==8.0.114
pandas==2.0.2
opencv-python==4.7.0.72
numpy==1.24.3
scipy==1.10.1
easyocr==1.7.0
filterpy==1.4.5

## Screenshots
<img width="1901" height="957" alt="Screenshot_11-6-2026_141226_127 0 0 1" src="https://github.com/user-attachments/assets/483c87a4-1607-4282-88d4-7a06e84a2f07" />


## Results
<img width="689" height="411" alt="image" src="https://github.com/user-attachments/assets/fae16cca-b2d4-44dc-acd1-d1b5f752d0cc" />
