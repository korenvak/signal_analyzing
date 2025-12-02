# Annotation System Implementation

This document describes the rectangle-based annotation system that has been added to the audio visualizer.

## Overview

The annotation system allows users to:
- Draw annotation rectangles directly on the spectrogram by mouse drag
- View annotations in a table panel below the spectrogram
- Edit annotation properties in the table
- Save and load annotations to/from JSON files
- Select and delete annotations

## Features

### Visual Representation

- **Red Border**: 2px width (2.5px when selected)
- **Semi-transparent Fill**: Red tint with 20% opacity (0.2 alpha) that tints but doesn't hide the spectrogram
- **Selection Highlighting**: Selected annotations have a thicker border

### Annotation Table

The table displays the following columns:
- **ID**: Unique annotation identifier
- **File Name**: Audio file basename
- **t_start**: Start time in seconds (read-only)
- **t_end**: End time in seconds (read-only)
- **f_min**: Minimum frequency in Hz (read-only)
- **f_max**: Maximum frequency in Hz (read-only)
- **Harmonic**: Harmonic index (editable)
- **SNR**: SNR estimate (editable)
- **Label**: Track label text (editable)
- **Approved**: Checkbox for approval status (editable)

### Mouse Interaction

- **Drawing**: Click and drag to create a new rectangle
- **Selection**: Click on an existing rectangle to select it
- **Editing**: Double-click table cells to edit

### Keyboard Shortcuts

- **A**: Toggle annotation mode on/off
- **Ctrl+Shift+S**: Save annotations
- **Delete**: Delete selected annotation

### Menu Items

The "Annotations" menu provides:
- Enable Annotation Mode (toggle)
- Save Annotations
- Load Annotations
- Delete Selected Annotation

## Architecture

### Components

1. **Annotation Data Model** (`annotation_data.py`)
   - `Annotation` dataclass representing a single annotation
   - Methods for serialization/deserialization

2. **Annotation Manager** (`annotation_manager.py`)
   - Manages collection of annotations for a file
   - Handles JSON save/load
   - Provides point-in-rectangle queries

3. **Annotation Renderer** (`annotation_renderer.py`)
   - Renders rectangle visuals using VisPy
   - Manages temporary rectangle during drawing
   - Handles selection highlighting

4. **Annotation Table** (`annotation_table.py`)
   - QTableWidget for displaying/editing annotations
   - Signals for selection, deletion, and updates

5. **Canvas Integration** (`vispy_canvas.py`)
   - Extended with annotation drawing mode
   - Mouse event handlers for drawing
   - Coordinate conversion (screen to world)

6. **Main Window Integration** (`main_window.py`)
   - Integrates all components
   - Connects signals and callbacks
   - Provides menu items

## Data Storage

Annotations are saved to JSON files with the naming pattern:
```
<audio_basename>_annotations.json
```

JSON structure:
```json
{
  "version": "1.0",
  "file_name": "sample_audio.wav",
  "annotations": [
    {
      "id": 1,
      "file_name": "sample_audio.wav",
      "t_start": 1.5,
      "t_end": 3.2,
      "f_min": 1000.0,
      "f_max": 5000.0,
      "harmonic_index": 1,
      "snr_estimate": 15.5,
      "track_label": "main track",
      "is_approved": true
    }
  ]
}
```

## Usage

1. **Enable Annotation Mode**: Press 'A' or use menu "Annotations > Enable Annotation Mode"

2. **Draw Rectangle**: Click and drag on the spectrogram to create a new annotation

3. **Select Annotation**: Click on an existing rectangle or select a row in the table

4. **Edit Properties**: Double-click editable cells in the table (Harmonic, SNR, Label, Approved)

5. **Delete Annotation**: Select and press Delete key, or use context menu

6. **Save Annotations**: Use "Annotations > Save Annotations" (Ctrl+Shift+S)

7. **Load Annotations**: Annotations are automatically loaded when opening an audio file if a matching JSON file exists, or use "Annotations > Load Annotations"

## Coordinate Conversion

The system correctly converts between:
- **Screen coordinates** (pixels): Mouse cursor position
- **World coordinates** (time, frequency): Spectrogram coordinate system

The conversion is handled by VisPy's transform system, ensuring annotations are always drawn at the correct time/frequency positions regardless of zoom or pan.

## Notes

- Annotations are stored in memory and linked between the table and visual rectangles by ID
- When loading a new audio file, annotations are cleared (file-specific)
- Annotations persist across zoom/pan operations
- The system supports one audio file at a time (no batch processing yet)

