# AI-based Video Editing from Description

A prototype that allows users to edit video clips based on natural language descriptions using AI models.

## Features

- **Object Removal/Replacement**: Remove or replace objects in videos (requires PyTorch + YOLO available)
- **Style/Scene Changes**: Apply visual effects (e.g., "Make it look like night-time")
- **Web-based UI**: Easy-to-use interface for uploading videos and entering edit descriptions

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

If PyTorch fails to install on Windows due to long-path issues, you can still run style edits (night-time, vintage, black & white). Object removal will be disabled until PyTorch/YOLO are installed.

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and go to the provided URL (usually http://localhost:7860)
3. Upload a video file (10-30 seconds recommended)
4. Enter your edit description in natural language
5. Click "Process Video" to generate the edited video

## Supported Edit Types

- **Object removal**: "Remove the [object] from the [location]"
- **Style changes**: "Make it look like night-time", "Apply vintage filter"
- **Scene modifications**: "Change the background to [description]"

## Technical Details

- Optionally uses YOLOv8 for object detection and segmentation (if installed)
- Implements inpainting for object removal
- Applies color grading and filters for style changes
- Processes videos frame by frame for consistent results
