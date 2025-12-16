# DAS Multi-Channel Waterfall System - Implementation Plan

## Overview

Add a new tab for DAS (Distributed Acoustic Sensing) multi-channel visualization while keeping the existing single-channel spectrogram functionality intact.

### Current vs New
| Aspect | Single Channel (Current) | DAS Multi-Channel (New) |
|--------|-------------------------|------------------------|
| Input | Audio file (FLAC/WAV) | Matrix files (float32) |
| Data | 1D audio → FFT → 2D | 2D matrix → direct display |
| X-axis | Time (seconds) | Sensors (integer IDs) |
| Y-axis | Frequency (Hz) | Time (HH:MM:SS) |
| Tab | "Single Channel" | "DAS Multi-Channel" |

---

## Architecture

### New File Structure
```
audio_visualizer/
├── core/
│   ├── data_source.py              # NEW: Abstract DataSource protocol
│   ├── das_data_provider.py        # NEW: DAS interface (USER IMPLEMENTS)
│   ├── time_formatter.py           # NEW: HH:MM:SS utilities
│   ├── waterfall_tile_manager.py   # NEW: Tile loading/caching
│   ├── waterfall_mipmap.py         # NEW: Multi-resolution pyramid
│   └── ... existing files
├── engines/
│   ├── waterfall_engine.py         # NEW: Matrix → display tiles
│   └── ... existing files
├── ui/
│   ├── main_window.py              # MODIFY: Add tab system
│   ├── das_tab.py                  # NEW: Main DAS container
│   ├── das_sidebar.py              # NEW: Sidebar with workflow tabs
│   ├── folder_selector.py          # NEW: Folder + metadata UI
│   ├── range_selector.py           # NEW: Sensor/time/freq selection
│   ├── waterfall_canvas.py         # NEW: Waterfall visualization
│   └── ... existing files
```

---

## UI Layout

### Main Window with Tabs
```
┌─────────────────────────────────────────────────────────────────┐
│  [Single Channel]  [DAS Multi-Channel]                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              (Tab content appears here)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DAS Tab Layout
```
┌────────────────┬────────────────────────────────────────────────┐
│    SIDEBAR     │                                                │
│   ┌──────────┐ │                                                │
│   │ 1.Folder │ │              WATERFALL DISPLAY                 │
│   │  Select  │ │                                                │
│   ├──────────┤ │      Sensors (X-axis, integers) →              │
│   │ 2.Range  │ │   T  ┌────────────────────────────────┐        │
│   │  Select  │ │   i  │                                │        │
│   ├──────────┤ │   m  │     [Waterfall Visualization]  │        │
│   │ 3.Display│ │   e  │                                │        │
│   │  Options │ │   ↓  │     Pan/Zoom/Scroll enabled    │        │
│   └──────────┘ │      └────────────────────────────────┘        │
│                │                                                │
│   [Compute]    │   Time format: HH:MM:SS                        │
│                │   Scroll: Navigate through time                │
└────────────────┴────────────────────────────────────────────────┘
```

### Sidebar Tab Details

**Tab 1: Folder Selection**
- Folder path input with browse button
- Metadata JSON file selector
- Display available data info:
  - Sensor range: [min] - [max]
  - Time ranges (showing gaps if any)
  - Sample rate
  - Data units
- "Load Metadata" button

**Tab 2: Range Selection**
- Sensor range: `[____]` to `[____]` (integers)
- Time range: `[HH:MM:SS]` to `[HH:MM:SS]`
- Frequency filter (optional): `[____]` Hz to `[____]` Hz
- Resolution preset: [Overview | Medium | Full]
- "Compute Waterfall" button

**Tab 3: Display Options**
- Colormap selector
- Normalization mode (MinMax / STD / Custom)
- dB scale toggle
- Filter controls (reuse existing filter system)
- Annotation tools

---

## Abstract Interfaces

### DASDataProvider Protocol (User Implements)
```python
# core/das_data_provider.py

from typing import Protocol, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import numpy as np

@dataclass
class DASMetadata:
    """Metadata loaded from JSON file"""
    sensor_range: Tuple[int, int]           # (min_sensor_id, max_sensor_id)
    time_ranges: List[Tuple[datetime, datetime]]  # Available ranges (gaps = missing entries)
    sample_rate: float                       # Samples per second
    sensor_spacing: Optional[float] = None   # Meters between sensors (optional)
    units: str = "phase_derivative"          # Data units description
    extra: dict = None                       # Additional metadata

@dataclass
class DASDataRequest:
    """Request parameters for data chunk"""
    sensor_start: int
    sensor_end: int
    time_start: datetime
    time_end: datetime
    downsample_factor: int = 1  # 1 = full res, 2 = half, etc.

class DASDataProvider(Protocol):
    """
    Abstract interface for DAS data access.
    USER IMPLEMENTS THIS for their specific file format.
    """

    def load_folder(self, folder_path: str, metadata_path: str) -> DASMetadata:
        """
        Load metadata and prepare for data access.

        Args:
            folder_path: Path to folder containing data files
            metadata_path: Path to JSON metadata file

        Returns:
            DASMetadata with available ranges
        """
        ...

    def get_metadata(self) -> DASMetadata:
        """Get currently loaded metadata"""
        ...

    def get_data_chunk(self, request: DASDataRequest) -> np.ndarray:
        """
        Get matrix chunk for specified range.

        Args:
            request: DASDataRequest with sensor/time ranges

        Returns:
            np.ndarray of shape (n_time_samples, n_sensors), dtype=float32
        """
        ...

    def get_data_chunk_async(
        self,
        request: DASDataRequest,
        callback: callable
    ) -> None:
        """
        Async version for background loading.

        Args:
            request: DASDataRequest
            callback: Function(data: np.ndarray, error: Optional[str])
        """
        ...

    def is_range_available(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start: datetime,
        time_end: datetime
    ) -> bool:
        """Check if requested range has data (no gaps)"""
        ...

    def close(self) -> None:
        """Release resources"""
        ...
```

### Mock Provider for Testing
```python
class MockDASDataProvider:
    """Mock provider for UI testing without real data"""

    def load_folder(self, folder_path: str, metadata_path: str) -> DASMetadata:
        return DASMetadata(
            sensor_range=(0, 2000),
            time_ranges=[
                (datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 1, 0, 0)),
                (datetime(2024, 1, 1, 2, 0, 0), datetime(2024, 1, 1, 4, 0, 0)),  # Gap!
            ],
            sample_rate=1000.0,
            sensor_spacing=1.0,
            units="rad/s"
        )

    def get_data_chunk(self, request: DASDataRequest) -> np.ndarray:
        n_sensors = request.sensor_end - request.sensor_start
        duration = (request.time_end - request.time_start).total_seconds()
        n_samples = int(duration * 1000.0 / request.downsample_factor)

        # Generate test pattern (gradient + noise)
        data = np.random.randn(n_samples, n_sensors).astype(np.float32) * 0.1
        # Add some structure for visual testing
        for i in range(n_sensors):
            data[:, i] += np.sin(np.linspace(0, 10*np.pi, n_samples) + i*0.1)

        return data
```

---

## Memory Management for Large Matrices

### Tile-Based Loading Strategy
```
Problem: 10,000 sensors × 3,600,000 samples (1 hour @ 1kHz) = 144 GB

Solution: Load only visible tiles + buffer

Tile Grid:
┌─────┬─────┬─────┬─────┬─────┐
│ 0,0 │ 1,0 │ 2,0 │ 3,0 │ ... │  ← Each tile: 512×512 = 1MB @ float32
├─────┼─────┼─────┼─────┼─────┤
│ 0,1 │ 1,1 │ 2,1 │ 3,1 │ ... │
├─────┼─────┼─────┼─────┼─────┤
│ ... │ ... │ ... │ ... │ ... │
└─────┴─────┴─────┴─────┴─────┘

Visible region loads:
- Visible tiles (immediate)
- 1-tile buffer around visible (prefetch)
- Async loading in background thread
```

### Multi-Resolution Pyramid (Mipmap)
```
Level 0: Full resolution     (1x)     - When zoomed in
Level 1: 2x downsampled      (1/2)
Level 2: 4x downsampled      (1/4)
Level 3: 8x downsampled      (1/8)
Level 4: 16x downsampled     (1/16)
Level 5: 32x downsampled     (1/32)   - Overview level

Zoom determines which level to use:
- zoom > 0.5: Level 0
- zoom 0.25-0.5: Level 1
- zoom 0.125-0.25: Level 2
- etc.
```

### GPU Memory Budget
```
Target: 1-2 GB for waterfall tiles

At 512×512 tiles, float32:
- 1 tile = 1 MB
- 64 tiles in GPU = 64 MB (visible + buffer)
- 256 tiles in CPU RAM = 256 MB (LRU cache)
- Disk cache = unlimited (zarr compressed)

Multi-resolution adds ~33% overhead (1 + 1/4 + 1/16 + ...)
```

---

## Waterfall Display Behavior

### Axes
- **X-axis**: Sensor ID (integers, e.g., 0-2000)
- **Y-axis**: Time (HH:MM:SS format)
  - Scrollable up/down
  - Shows relative or absolute time

### Navigation
- **Mouse wheel**: Scroll through time (Y-axis)
- **Click + drag**: Pan in both directions
- **Ctrl + wheel**: Zoom in/out
- **Zoom**: Changes resolution level automatically

### Time Formatting
```python
# core/time_formatter.py

class TimeAxisFormatter:
    def __init__(self, base_time: datetime, sample_rate: float):
        self.base_time = base_time
        self.sample_rate = sample_rate

    def sample_to_time(self, sample_idx: int) -> datetime:
        seconds = sample_idx / self.sample_rate
        return self.base_time + timedelta(seconds=seconds)

    def format_display(self, sample_idx: int, show_ms: bool = False) -> str:
        t = self.sample_to_time(sample_idx)
        if show_ms:
            return t.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        return t.strftime("%H:%M:%S")

    def parse_input(self, time_str: str) -> int:
        """Parse user input HH:MM:SS to sample index"""
        t = datetime.strptime(time_str, "%H:%M:%S")
        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        return int(delta.total_seconds() * self.sample_rate)
```

---

## Feature Compatibility

### Annotations
| Feature | Spectrogram | Waterfall | Notes |
|---------|-------------|-----------|-------|
| Rectangle annotations | ✓ | ✓ | Different coord system |
| Curve/track drawing | ✓ | ✓ | Sensor vs freq curves |
| Labels | ✓ | ✓ | Same |
| Colors | ✓ | ✓ | Same |
| JSON storage | ✓ | ✓ | Add coord_system field |

**Annotation Data Extension**:
```python
@dataclass
class Annotation:
    # ... existing fields ...
    coord_system: str = 'spectrogram'  # 'spectrogram' or 'waterfall'
    # For spectrogram: x=time(s), y=freq(Hz)
    # For waterfall: x=sensor_id, y=time(sample_idx or datetime)
```

### Filters
| Filter | Spectrogram | Waterfall | Notes |
|--------|-------------|-----------|-------|
| Gaussian blur | ✓ | ✓ | Universal |
| Median filter | ✓ | ✓ | Universal |
| Contrast enhance | ✓ | ✓ | Universal |
| CLAHE | ✓ | ✓ | Universal |
| Morphological | ✓ | ✓ | Universal |
| Bilateral | ✓ | ✓ | Universal |
| Total variation | ✓ | ✓ | Universal |
| Non-local means | ✓ | ✓ | Universal |
| Spectral subtraction | ✓ | ✗ | Freq-specific |
| Harmonic/Percussive | ✓ | ✗ | Audio-specific |
| PCEN | ✓ | ✗ | Spectrogram-specific |

**Filter Manager Extension**:
```python
# Add to filter definitions
FILTER_COMPATIBILITY = {
    'gaussian_blur': ['spectrogram', 'waterfall'],
    'median_filter': ['spectrogram', 'waterfall'],
    'spectral_subtraction': ['spectrogram'],  # Only spectrogram
    'harmonic_percussive': ['spectrogram'],
    # ... etc
}
```

### Other Features
| Feature | Spectrogram | Waterfall | Notes |
|---------|-------------|-----------|-------|
| Cutout extraction | ✓ | ✓ | Works on any 2D region |
| Export (image) | ✓ | ✓ | Same |
| Export (numpy) | ✓ | ✓ | Same |
| Colormap | ✓ | ✓ | Same |
| Event markers | ✓ | ✓ | Vertical lines at time |
| Measurement tool | ✓ | ✓ | Different units |
| Doppler detection | ✓ | ✗ | Freq-specific |

---

## Implementation Phases

### Phase 1: Foundation (No UI changes yet)
- [ ] Create `core/data_source.py` - Abstract protocols
- [ ] Create `core/das_data_provider.py` - DAS interface + mock
- [ ] Create `core/time_formatter.py` - Time utilities
- [ ] Extend `annotation_data.py` - Add coord_system field
- [ ] Extend `filter_manager.py` - Add compatibility flags

### Phase 2: Tab System
- [ ] Modify `main_window.py` - Add QTabWidget
- [ ] Move existing UI to "Single Channel" tab
- [ ] Create placeholder "DAS Multi-Channel" tab
- [ ] Test: All existing functionality works

### Phase 3: DAS UI Shell
- [ ] Create `ui/das_tab.py` - Main container
- [ ] Create `ui/das_sidebar.py` - Sidebar with sub-tabs
- [ ] Create `ui/folder_selector.py` - Folder selection widget
- [ ] Create `ui/range_selector.py` - Range selection widget
- [ ] Wire up basic signals/slots

### Phase 4: Waterfall Engine
- [ ] Create `core/waterfall_tile_manager.py` - Tile management
- [ ] Create `core/waterfall_mipmap.py` - Multi-resolution
- [ ] Create `engines/waterfall_engine.py` - Processing
- [ ] Implement GPU texture streaming

### Phase 5: Waterfall Canvas
- [ ] Create `ui/waterfall_canvas.py` - VisPy-based display
- [ ] Implement tiled rendering
- [ ] Implement pan/scroll navigation
- [ ] Implement zoom with resolution switching
- [ ] Implement time axis (HH:MM:SS)
- [ ] Implement sensor axis (integers)

### Phase 6: Feature Integration
- [ ] Connect annotation system to waterfall
- [ ] Connect filter system to waterfall
- [ ] Adapt cutout extraction
- [ ] Adapt measurement tool

### Phase 7: Polish
- [ ] Performance optimization
- [ ] Memory leak testing
- [ ] Error handling
- [ ] Loading indicators
- [ ] User feedback

---

## Technical Specifications

### Constants
```python
# Tile configuration
WATERFALL_TILE_SIZE = (512, 512)  # (sensors, time_samples)
MAX_GPU_TILES = 64                 # ~64 MB GPU memory
MAX_CPU_TILES = 256                # ~256 MB RAM
MIPMAP_LEVELS = 6                  # 1x to 32x downsampling

# Display
TIME_FORMAT = "%H:%M:%S"
TIME_FORMAT_MS = "%H:%M:%S.%f"
DEFAULT_COLORMAP = "viridis"

# Performance
PREFETCH_TILES = 1                 # Tiles to prefetch around visible
ASYNC_LOAD_THREADS = 4             # Background loader threads
```

### Dependencies (No new dependencies needed)
- NumPy (existing)
- VisPy (existing)
- PyQt5/PySide6 (existing)
- CuPy (existing, optional)
- Zarr (existing, for disk cache)

---

## Notes for User Implementation

When you implement `DASDataProvider`:

1. **load_folder()**: Parse your metadata JSON and index available files
2. **get_data_chunk()**: Read from your file format, return `(time, sensors)` float32 array
3. **Handle gaps**: Your metadata should list discontinuous time ranges
4. **Downsample efficiently**: If `downsample_factor > 1`, consider reading strided or pre-downsampling

Example metadata JSON structure you might use:
```json
{
  "sensor_range": [0, 2048],
  "sample_rate": 1000.0,
  "sensor_spacing_m": 1.02,
  "units": "rad/s",
  "files": [
    {
      "filename": "data_001.bin",
      "time_start": "2024-01-15T10:00:00",
      "time_end": "2024-01-15T10:30:00",
      "sensors": [0, 2048]
    },
    {
      "filename": "data_002.bin",
      "time_start": "2024-01-15T10:45:00",
      "time_end": "2024-01-15T11:15:00",
      "sensors": [0, 2048]
    }
  ]
}
```
