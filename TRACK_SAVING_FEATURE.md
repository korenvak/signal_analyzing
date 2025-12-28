# Track Saving Feature

## Overview

You can now save frequency tracks that you paint over the spectrogram with full interpolation, amplitude sampling, and metadata export.

## Features

### 1. **Monotone Cubic Interpolation (PCHIP)**
- Uses `scipy.interpolate.PchipInterpolator` for smooth, non-oscillating curves
- Preserves monotonicity between control points
- Generates straight/curved lines instead of polynomial oscillations
- Automatically removes duplicate time values

### 2. **Data Exports**

When you save a track, the system creates:

#### **Summary CSV** (`tracks.csv`)
Contains one row per track with:
- `id`: Track identifier
- `audio_file`: Source audio filename
- `pixel`: Sensor/pixel ID (parsed from filename)
- `t_start`, `t_end`: Start and end times (seconds)
- `f_min`, `f_max`: Minimum and maximum frequencies (Hz)
- `n_control_points`: Number of user-clicked points
- `n_interpolated_points`: Number of interpolated samples
- `control_points`: Raw control points data (format: "t1,f1;t2,f2;...")
- `track_start_absolute`: Absolute start timestamp (from filename)
- `track_end_absolute`: Absolute end timestamp (from filename)
- `track_label`: User-provided label
- `notes`: User notes
- `image_path`: Path to PNG export
- `created_at`: Creation timestamp

#### **Detailed Data CSV** (`track_data/track_NNNN_data.csv`)
Contains interpolated data points with:
- `track_id`: Track identifier
- `audio_file`: Source audio filename
- `pixel`: Sensor/pixel ID
- `time_seconds`: Interpolated time value
- `frequency_hz`: Interpolated frequency value
- `amplitude_db`: Amplitude/dB sampled from spectrogram at this point

#### **PNG Image** (`track_images/track_NNNN_YYYYMMDD_HHMMSS.png`)
High-resolution spectrogram image showing:
- Spectrogram background with your chosen colormap
- Yellow interpolated curve overlay
- Cyan control points (user clicks)
- Legend, colorbar, and metadata in title
- Properly labeled axes with time and frequency

### 3. **Metadata Captured**

The system automatically captures:

1. **Amplitude/dB values**: Sampled from the spectrogram at each interpolated point
2. **Frequency**: Interpolated frequency values along the track
3. **Time**: Interpolated time values along the track
4. **PNG export**: Visual representation with track overlay
5. **Metadata**:
   - Source audio filename
   - Pixel/sensor ID (if parseable from filename format like `pixel_1008_...`)
   - Absolute timestamps (if filename contains timestamp)
   - Track bounds (time and frequency ranges)
   - Number of control points and interpolated points
   - Creation timestamp

## How to Use

### Step 1: Enter Curve Mode
Press **`C`** key to enter curve drawing mode.

### Step 2: Paint Your Track
- **Left-click** on the spectrogram to add control points
- The curve will update in real-time with smooth interpolation
- **Right-click** to clear the current track
- **Minimum 2 points required** (status bar shows how many more you need)

### Step 3: Finish and Save
Press **`Enter`** when done drawing.

A dialog will ask if you want to save the track:
- Click **Yes** to save with all exports
- Click **No** to skip saving (but curve will still be saved to annotation if one is selected)

### Step 4: Enter Track Label
If you chose to save, you'll be prompted for an optional track label.

### Step 5: Confirmation
A success dialog shows:
- Track ID
- Number of control points and interpolated points
- Duration and frequency range
- Paths to all created files

## File Organization

When you save your first track for an audio file, a session directory is created:

```
<audio_filename>_tracks/
├── tracks.csv                    # Summary CSV with all tracks
├── track_data/                   # Detailed interpolated data
│   ├── track_0001_data.csv
│   ├── track_0002_data.csv
│   └── ...
└── track_images/                 # PNG exports
    ├── track_0001_20250128_143022.png
    ├── track_0002_20250128_143045.png
    └── ...
```

## Example Workflow

1. Load an audio file with DAS/pixel data (e.g., `pixel_1008_20250115_120000_20250115_120030.wav`)
2. View the spectrogram
3. Press **`C`** to enter curve mode
4. Click along a frequency track in the spectrogram
5. Press **`Enter`** when done
6. Click **Yes** to save the track
7. Enter a label like "Doppler Event 1"
8. Check the created `<filename>_tracks/` directory for all exports

## CSV Format Examples

### Summary CSV (`tracks.csv`)
```csv
id,audio_file,pixel,t_start,t_end,f_min,f_max,n_control_points,n_interpolated_points,control_points,track_start_absolute,track_end_absolute,track_label,notes,image_path,created_at
1,pixel_1008_20250115_120000_20250115_120030.wav,1008,5.234567,12.345678,1200.00,3400.00,8,200,"5.234567,1200.00;6.123456,1500.00;...",2025-01-15 12:00:05.234,2025-01-15 12:00:12.345,Doppler Event 1,,C:\...\track_images\track_0001_20250128_143022.png,2025-01-28 14:30:22.123
```

### Detailed Data CSV (`track_0001_data.csv`)
```csv
track_id,audio_file,pixel,time_seconds,frequency_hz,amplitude_db
1,pixel_1008_20250115_120000_20250115_120030.wav,1008,5.234567,1200.00,-45.23
1,pixel_1008_20250115_120000_20250115_120030.wav,1008,5.271234,1215.43,-44.87
1,pixel_1008_20250115_120000_20250115_120030.wav,1008,5.307901,1230.89,-44.52
...
```

## Technical Details

### Interpolation Algorithm
- **Method**: PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
- **Why PCHIP?**:
  - Monotone cubic interpolation prevents oscillations
  - Produces smooth curves without overshoot
  - Better for frequency tracks than regular cubic splines
  - Handles non-uniform sampling well
- **Fallback**: Linear interpolation if PCHIP unavailable

### Amplitude Sampling
- Uses nearest-neighbor sampling in both time and frequency dimensions
- Samples the spectrogram's dB values at each interpolated point
- Handles out-of-bounds gracefully with NaN values

### Filename Parsing
- Automatically parses pixel/sensor ID from filenames like `pixel_NNNN_...`
- Extracts timestamps from filename format: `pixel_1008_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.wav`
- Computes absolute timestamps for track start/end if filename is parseable

## Integration with Annotations

The track painting feature integrates seamlessly with the existing annotation system:

- If you have an annotation selected when you finish drawing a curve, the curve points are automatically saved to that annotation for Doppler analysis
- The track saving is independent - you can save tracks without annotations
- If you have 4+ points, Doppler analysis is automatically triggered for the selected annotation

## Keyboard Shortcuts

- **`C`**: Toggle curve drawing mode
- **Left-click**: Add control point
- **Right-click**: Clear current curve
- **`Enter`**: Finish curve and save

## Notes

- The interpolation uses 200 samples by default for smooth curves
- PNG images are saved at 150 DPI for good quality
- Track sessions are per-audio-file (each audio file gets its own track directory)
- All CSV files use UTF-8 encoding
- Image export requires matplotlib
- PCHIP interpolation requires scipy

## Future Enhancements

Possible improvements:
- Adjustable number of interpolation samples
- Multi-track visualization overlay
- Track comparison tools
- Export to other formats (JSON, MAT, HDF5)
- Batch processing of multiple tracks
