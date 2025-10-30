Medical Prescription Analyzer

A free, open-source Python app that analyzes uploaded prescription images (printed or handwritten) using AI vision. It extracts key details (patient name, date, medications, doctor), fetches real-time drug insights (side effects via OpenFDA), and generates a markdown report. Built for educational purposes—not medical advice. Always consult professionals.
Features

Vision Extraction: Uses Google's Gemini 2.5 Flash (free API) for OCR and structured JSON parsing from images.
Agentic Workflow: LangGraph ReAct agent autonomously fetches drug data (e.g., side effects) for each medication.
User-Friendly UI: Gradio web interface for easy upload and real-time results (JSON + markdown report).
Error-Resilient: Handles blurry/handwritten text, API errors, and empty results gracefully.
Zero-Cost MVP: Runs locally; free APIs only (Gemini + OpenFDA).
Modular & Extensible: Easy to add tools (e.g., GoodRx for prices) or deploy to Hugging Face Spaces.

Demo

Upload a prescription image.
Click "Analyze" → See extracted JSON, insights, and full report.

(Add a GIF/screenshot here in your repo: e.g., via GitHub's image upload. Example: Failed to load imageView link)
Sample Output (JSON):
json{
  "patient_name": "John Doe",
  "date": "2025-10-30",
  "medications": [
    {
      "name": "Aspirin",
      "dosage": "100mg",
      "frequency": "daily"
    }
  ],
  "doctor_name": "Dr. Smith"
}
Report Snippet:
text# Prescription Analysis Report

## Extracted Details
{JSON above}

## Drug Insights
### Aspirin
Side effects: Headache, nausea... Price estimate: Check GoodRx
Quick Start
Prerequisites

Python 3.10+.
Git installed.
Free API keys (see below).

Setup

Clone the Repo:
textgit clone https://github.com/YOUR_USERNAME/medical-prescription-analyzer.git
cd medical-prescription-analyzer

Virtual Environment:
textpython -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

Install Dependencies:
textpip install -r requirements.txt

API Keys (Add to .env—create from .env.example):

Gemini API (for vision extraction): Free at Google AI Studio. Add GEMINI_API_KEY=your_key.
OpenFDA API (for drug info): Free at open.fda.gov. Optional key for higher limits: FDA_API_KEY=your_key.
Security: .env is in .gitignore—never commit keys!


Run the App:
textpython Main.py  # Or app.py if renamed

Opens at http://127.0.0.1:7860. Upload and analyze!

Usage

Test Images: Use free samples from Kaggle ("prescription images dataset") or print a mock one.
Handwriting: Works best on clear scans; for tough cases, enhance image contrast pre-upload.
Customization: Edit Main.py prompts for better accuracy (e.g., add domain-specific terms).

Tech Stack

Frontend: Gradio (web UI).
AI Core: LangChain + LangGraph (agentic flows), Google Gemini 2.5 Flash (free vision LLM).
Tools: Requests (APIs), Pillow (images), Re (JSON parsing).
Version Control: Git/GitHub—forked inspirations from kingabzpro/Medical-AI-with-Grok4 and Shriram2005/MediScribe-OCR.

API Setup Guide

Gemini:

Visit AI Studio.
Create key → Select "Gemini 2.5 Flash" → Copy to .env.
Limits: 15 RPM, 1M tokens/day (free).

OpenFDA:

No key needed for basics; register at open.fda.gov for more.
Test: https://api.fda.gov/drug/label.json?search=brand_name:"aspirin".


Deployment

Local: As above.
Free Hosting: Push to Hugging Face Spaces—auto-deploys Gradio.
Docker: Add Dockerfile for containerization (e.g., FROM python:3.12-slim).