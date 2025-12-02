# Acoustic Analysis & Doppler Detection - Technical Roadmap

## Document Purpose
This document outlines the complete technical plan for improving the acoustic spectrogram analysis application, specifically focused on Doppler effect detection, track analysis, and research-grade data export. It serves as a reference for any developer or LLM system to understand the current state, problems, and implementation strategies.

---

## Table of Contents
1. [Current Application State](#1-current-application-state)
2. [Priority 1: Data Management & Research Export](#2-priority-1-data-management--research-export)
3. [Priority 2: Track Analysis Improvements](#3-priority-2-track-analysis-improvements)
4. [Priority 3: Harmonic Detection](#4-priority-3-harmonic-detection)
5. [Priority 4: Advanced Export & Event Grouping](#5-priority-4-advanced-export--event-grouping)
6. [Priority 5: Automatic Track Detection](#6-priority-5-automatic-track-detection)
7. [Priority 6: GPS Integration (Future)](#7-priority-6-gps-integration-future)
8. [Priority 7: Doppler Model Limitations](#8-priority-7-doppler-model-limitations)
9. [Reference: Existing Detection Algorithm Analysis](#9-reference-existing-detection-algorithm-analysis)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Current Application State

### 1.1 What Works
- **GPU-accelerated spectrogram visualization** using VisPy and CuPy
- **Annotation system** with rectangular regions on spectrogram
- **Doppler curve drawing** - users can draw polyline curves on annotations
- **Measurement tool** - continuous point-to-point measurements with visual feedback
- **Annotation table** with visibility toggles (View, Curve columns)
- **Basic Doppler analysis** - curve fitting to extract velocity (has limitations, see Priority 7)
- **FFT spectrum dialog** - view frequency content of selected regions
- **Settings menu** - FFT size, overlap, window function with immediate update
- **Annotation persistence** - JSON-based save/load per audio file

### 1.2 Key Files
```
audio_visualizer/
├── core/
│   ├── doppler_analysis.py      # Doppler curve fitting (needs GPS for accuracy)
│   ├── data_loader.py           # Audio file loading
│   └── spectrogram_engine.py    # FFT computation
├── ui/
│   ├── main_window.py           # Main application window
│   ├── vispy_canvas.py          # Spectrogram visualization & interaction
│   ├── annotation_data.py       # Annotation data model
│   ├── annotation_manager.py    # Annotation collection management
│   ├── annotation_renderer.py   # Visual rendering of annotations
│   ├── annotation_table.py      # Table widget for annotations
│   ├── measurement_panel.py     # Measurement history panel
│   └── spectrum_dialog.py       # FFT spectrum viewer
```

### 1.3 Current Data Model (annotation_data.py)
```python
@dataclass
class Annotation:
    id: int
    file_name: str
    t_start: float      # Start time (seconds)
    t_end: float        # End time (seconds)
    f_min: float        # Min frequency (Hz)
    f_max: float        # Max frequency (Hz)
    
    # Visibility
    is_visible: bool = True
    show_doppler_curve: bool = True
    
    # Doppler data
    points: List[Tuple[float, float]] = []  # User-drawn curve [(t, f), ...]
    doppler_result: Optional[Dict] = None   # Fit results (currently unreliable)
    
    # Labels
    track_label: str = ""
    
    # Harmonic linking (partially implemented)
    harmonic_order: Optional[int] = None
    parent_rect_id: Optional[int] = None
    event_id: Optional[int] = None
```

---

## 2. Priority 1: Data Management & Research Export

### 2.1 Problem Statement
Current data storage is fragmented:
- Annotations saved per-file as `{filename}_annotations.json`
- No unified project structure
- Spectrogram parameters not saved with annotations
- Cannot reproduce exact analysis conditions
- No multi-file aggregation for research

### 2.2 Proposed Solution: Project-Based Architecture

#### 2.2.1 Directory Structure
```
project_folder/
├── project.json                    # Master metadata
├── files/
│   ├── audio_file_1/
│   │   ├── file_metadata.json      # Per-file analysis parameters
│   │   ├── annotations.json        # All annotations for this file
│   │   ├── tracks/                 # Detailed track data
│   │   │   ├── track_001.json
│   │   │   └── track_002.json
│   │   └── cutouts/                # Exported spectrograms
│   │       ├── annotation_1.png
│   │       └── annotation_1_meta.json
│   └── audio_file_2/
│       └── ...
├── events/
│   └── grouped_events.json         # Cross-file event grouping
└── exports/
    ├── all_annotations.csv         # Unified CSV export
    ├── all_events.csv              # Event-level summary
    └── analysis_report.html        # Visual report
```

#### 2.2.2 project.json Schema
```json
{
  "version": "2.0",
  "created": "2025-06-02T10:30:00Z",
  "modified": "2025-06-02T15:45:00Z",
  "name": "Field Recording Analysis - Site A",
  "description": "Analysis of acoustic events from deployment 2025-06",
  
  "files": [
    {
      "filename": "recording_001.wav",
      "path": "files/recording_001/",
      "duration_seconds": 3600.0,
      "sample_rate": 44100,
      "analyzed": true,
      "annotation_count": 15,
      "event_count": 5
    }
  ],
  
  "global_settings": {
    "default_fft_size": 4096,
    "default_hop_length": 512,
    "default_window": "blackman",
    "default_overlap_percent": 87.5
  },
  
  "sensor_info": {
    "sensor_id": "sensor_001",
    "gps_file": null,
    "location_description": "Site A - North corner"
  }
}
```

#### 2.2.3 file_metadata.json Schema
```json
{
  "filename": "recording_001.wav",
  "original_path": "C:/recordings/recording_001.wav",
  "file_hash": "sha256:abc123...",
  
  "audio_properties": {
    "sample_rate": 44100,
    "duration_seconds": 3600.0,
    "channels": 1,
    "bit_depth": 16
  },
  
  "analysis_parameters": {
    "fft_size": 4096,
    "hop_length": 512,
    "window_function": "blackman",
    "overlap_percent": 87.5,
    "freq_range_analyzed": [0, 22050]
  },
  
  "analysis_history": [
    {
      "timestamp": "2025-06-02T10:30:00Z",
      "action": "initial_analysis",
      "parameters": {...}
    }
  ]
}
```

#### 2.2.4 annotations.json Schema (Enhanced)
```json
{
  "file": "recording_001.wav",
  "spectrogram_params": {
    "fft_size": 4096,
    "hop_length": 512,
    "window": "blackman"
  },
  
  "annotations": [
    {
      "id": 1,
      "bounds": {
        "t_start": 125.340,
        "t_end": 132.890,
        "f_min": 450.0,
        "f_max": 2200.0
      },
      
      "track": {
        "points": [[125.5, 1850.0], [126.0, 1720.0], ...],
        "point_count": 25,
        "duration_seconds": 7.55,
        "freq_range": [520.0, 1850.0]
      },
      
      "analysis": {
        "snr_db": 18.5,
        "snr_method": "adaptive_bandwidth",
        "bandwidth_profile": {
          "method": "fwhm",
          "values": [[125.5, 45.0], [126.0, 42.0], ...]
        },
        "slope_hz_per_sec": -175.3,
        "slope_variation": 12.4
      },
      
      "harmonic_info": {
        "harmonic_order": 2,
        "estimated_f0": 925.0,
        "parent_annotation_id": null,
        "linked_harmonics": [3, 4, 5],
        "event_id": "EVT_001"
      },
      
      "metadata": {
        "created": "2025-06-02T10:35:00Z",
        "modified": "2025-06-02T11:20:00Z",
        "label": "vehicle_pass",
        "notes": "Clear S-curve, likely truck"
      },
      
      "visibility": {
        "is_visible": true,
        "show_curve": true
      }
    }
  ]
}
```

### 2.3 Implementation Steps

#### Step 2.3.1: Create ProjectManager Class
**File:** `audio_visualizer/core/project_manager.py`

```python
class ProjectManager:
    """Manages project-level data and file organization."""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        self.project_data = {}
        self.file_managers = {}  # filename -> AnnotationManager
    
    def create_project(self, path: Path, name: str) -> bool:
        """Create new project structure."""
        pass
    
    def load_project(self, path: Path) -> bool:
        """Load existing project."""
        pass
    
    def save_project(self) -> bool:
        """Save all project data."""
        pass
    
    def add_file(self, audio_path: Path, params: dict) -> str:
        """Add audio file to project."""
        pass
    
    def get_file_annotations(self, filename: str) -> List[Annotation]:
        """Get annotations for specific file."""
        pass
    
    def export_unified_csv(self, output_path: Path) -> bool:
        """Export all annotations across all files to CSV."""
        pass
    
    def export_events_csv(self, output_path: Path) -> bool:
        """Export grouped events to CSV."""
        pass
```

#### Step 2.3.2: Update AnnotationManager
**File:** `audio_visualizer/ui/annotation_manager.py`

Add methods:
- `save_with_metadata(params: dict)` - Save with spectrogram parameters
- `load_with_validation(audio_path: Path)` - Validate file exists
- `get_analysis_summary() -> dict` - Return statistics

#### Step 2.3.3: Update MainWindow
**File:** `audio_visualizer/ui/main_window.py`

Add:
- Project menu (New Project, Open Project, Save Project)
- Track spectrogram parameters with each annotation save
- Auto-save on file switch

---

## 3. Priority 2: Track Analysis Improvements

### 3.1 Problem Statement
Current track analysis limitations:
1. **No SNR calculation** - Cannot quantify signal quality
2. **Track has no width** - Real signals have frequency spread
3. **Width varies with frequency** - Higher frequencies = narrower bandwidth
4. **No slope analysis** - Important for Doppler characterization

### 3.2 Proposed Solution: Adaptive Bandwidth SNR

#### 3.2.1 Bandwidth Model
For acoustic signals, bandwidth typically follows:
```
bandwidth(f) = f / Q
```
Where Q (quality factor) is approximately constant for a given source type.

Alternative: Use FWHM (Full Width at Half Maximum) measured from spectrogram.

#### 3.2.2 SNR Calculation Algorithm
```python
def calculate_track_snr(spectrogram: np.ndarray, 
                        track_points: List[Tuple[float, float]],
                        times: np.ndarray,
                        freqs: np.ndarray,
                        q_factor: float = 10.0) -> dict:
    """
    Calculate SNR for a track with adaptive bandwidth.
    
    Args:
        spectrogram: 2D array (freq x time) in dB
        track_points: List of (time, freq) points
        times: Time axis array
        freqs: Frequency axis array
        q_factor: Q factor for bandwidth calculation
    
    Returns:
        dict with snr_db, bandwidth_profile, signal_power, noise_power
    """
    signal_power = []
    noise_power = []
    bandwidth_profile = []
    
    for t, f in track_points:
        # Calculate bandwidth at this frequency
        bandwidth = f / q_factor
        
        # Find indices
        t_idx = np.argmin(np.abs(times - t))
        f_idx = np.argmin(np.abs(freqs - f))
        
        # Bandwidth in bins
        df = freqs[1] - freqs[0]
        bw_bins = int(bandwidth / df / 2)
        
        # Signal region: within bandwidth of track
        f_min_idx = max(0, f_idx - bw_bins)
        f_max_idx = min(len(freqs), f_idx + bw_bins)
        signal_region = spectrogram[f_min_idx:f_max_idx, t_idx]
        
        # Noise region: same width, but offset above and below
        noise_offset = bw_bins * 3  # Gap between signal and noise
        
        noise_above_min = f_max_idx + noise_offset
        noise_above_max = noise_above_min + (f_max_idx - f_min_idx)
        
        noise_below_max = f_min_idx - noise_offset
        noise_below_min = noise_below_max - (f_max_idx - f_min_idx)
        
        noise_regions = []
        if noise_above_max < len(freqs):
            noise_regions.append(spectrogram[noise_above_min:noise_above_max, t_idx])
        if noise_below_min >= 0:
            noise_regions.append(spectrogram[noise_below_min:noise_below_max, t_idx])
        
        if noise_regions:
            noise_region = np.concatenate(noise_regions)
        else:
            noise_region = np.array([spectrogram.min()])
        
        # Power calculation (convert from dB)
        sig_power = np.mean(10 ** (signal_region / 10))
        nse_power = np.mean(10 ** (noise_region / 10))
        
        signal_power.append(sig_power)
        noise_power.append(nse_power)
        bandwidth_profile.append((t, f, bandwidth))
    
    # Overall SNR
    total_signal = np.mean(signal_power)
    total_noise = np.mean(noise_power)
    snr_db = 10 * np.log10(total_signal / total_noise) if total_noise > 0 else float('inf')
    
    return {
        'snr_db': snr_db,
        'bandwidth_profile': bandwidth_profile,
        'signal_power_mean': total_signal,
        'noise_power_mean': total_noise,
        'q_factor_used': q_factor
    }
```

#### 3.2.3 Slope Analysis
```python
def analyze_track_slope(track_points: List[Tuple[float, float]]) -> dict:
    """
    Analyze the slope characteristics of a Doppler track.
    
    Returns:
        dict with overall_slope, slope_at_points, slope_variation
    """
    times = np.array([p[0] for p in track_points])
    freqs = np.array([p[1] for p in track_points])
    
    # Overall slope (linear fit)
    slope, intercept = np.polyfit(times, freqs, 1)
    
    # Local slopes (for each segment)
    local_slopes = []
    for i in range(len(times) - 1):
        dt = times[i+1] - times[i]
        df = freqs[i+1] - freqs[i]
        if dt > 0:
            local_slopes.append(df / dt)
    
    return {
        'overall_slope_hz_per_sec': slope,
        'intercept_hz': intercept,
        'local_slopes': local_slopes,
        'slope_mean': np.mean(local_slopes) if local_slopes else 0,
        'slope_std': np.std(local_slopes) if local_slopes else 0,
        'max_slope': max(local_slopes) if local_slopes else 0,
        'min_slope': min(local_slopes) if local_slopes else 0
    }
```

### 3.3 Implementation Steps

#### Step 3.3.1: Create TrackAnalyzer Class
**File:** `audio_visualizer/core/track_analyzer.py`

```python
class TrackAnalyzer:
    """Analyzes Doppler tracks for SNR, slope, and other metrics."""
    
    def __init__(self, q_factor: float = 10.0):
        self.q_factor = q_factor
    
    def analyze(self, spectrogram: np.ndarray, 
                track_points: List[Tuple[float, float]],
                times: np.ndarray, 
                freqs: np.ndarray) -> TrackAnalysisResult:
        """Full track analysis."""
        pass
    
    def calculate_snr(self, ...) -> float:
        pass
    
    def calculate_slope(self, ...) -> dict:
        pass
    
    def estimate_bandwidth_profile(self, ...) -> List[Tuple]:
        pass
```

#### Step 3.3.2: Update Annotation Data Model
Add to `Annotation` class:
```python
# Analysis results
snr_db: Optional[float] = None
slope_hz_per_sec: Optional[float] = None
bandwidth_profile: Optional[List[Tuple]] = None
analysis_params: Optional[Dict] = None  # Q factor, method, etc.
```

#### Step 3.3.3: Integrate with UI
- Add "Analyze Track" button/menu option
- Display SNR in annotation table
- Visualize bandwidth "tube" around track (optional)

---

## 4. Priority 3: Harmonic Detection

### 4.1 Problem Statement
Doppler signals often contain multiple harmonics:
- Fundamental frequency f0
- Harmonics at 2*f0, 3*f0, 4*f0, etc.
- Sub-harmonics at f0/2, f0/3 (less common)

Currently:
- User must manually identify and link harmonics
- No automatic detection
- No f0 estimation from harmonic relationships

### 4.2 Proposed Solution: Harmonic Search Algorithm

#### 4.2.1 Algorithm Overview
1. User marks one track (any harmonic)
2. System estimates possible f0 values
3. System searches for tracks at harmonic frequencies
4. Correlation-based matching considering:
   - Slope scales with frequency: `slope_n = n * slope_1`
   - Time alignment (same event = same time window)
   - Shape similarity

#### 4.2.2 Implementation
```python
class HarmonicDetector:
    """Detects and links harmonic tracks."""
    
    def __init__(self, spectrogram: np.ndarray, times: np.ndarray, freqs: np.ndarray):
        self.spectrogram = spectrogram
        self.times = times
        self.freqs = freqs
    
    def find_harmonics(self, 
                       seed_track: List[Tuple[float, float]],
                       max_harmonic: int = 8,
                       correlation_threshold: float = 0.7) -> List[HarmonicMatch]:
        """
        Find harmonic tracks related to a seed track.
        
        Args:
            seed_track: The user-marked track points
            max_harmonic: Maximum harmonic number to search
            correlation_threshold: Minimum correlation for match
        
        Returns:
            List of HarmonicMatch objects with track points and confidence
        """
        # Estimate f0 candidates
        f0_candidates = self._estimate_f0(seed_track)
        
        matches = []
        for f0 in f0_candidates:
            for n in range(1, max_harmonic + 1):
                if n == 1 and abs(f0 - np.median([p[1] for p in seed_track])) < 50:
                    continue  # Skip if this is the seed track itself
                
                # Expected frequency for harmonic n
                expected_track = self._scale_track(seed_track, n, f0)
                
                # Search for actual track near expected
                found_track = self._search_track(expected_track)
                
                if found_track:
                    correlation = self._calculate_correlation(expected_track, found_track)
                    if correlation > correlation_threshold:
                        matches.append(HarmonicMatch(
                            harmonic_order=n,
                            f0=f0,
                            track_points=found_track,
                            correlation=correlation
                        ))
        
        return matches
    
    def _estimate_f0(self, track: List[Tuple[float, float]]) -> List[float]:
        """Estimate possible f0 values from track."""
        median_freq = np.median([p[1] for p in track])
        
        # Track could be harmonic 1, 2, 3, etc.
        candidates = []
        for n in range(1, 6):
            f0 = median_freq / n
            if f0 > 20:  # Minimum audible frequency
                candidates.append(f0)
        
        return candidates
    
    def _scale_track(self, track: List[Tuple[float, float]], 
                     harmonic_n: int, f0: float) -> List[Tuple[float, float]]:
        """Scale track to expected harmonic frequency."""
        seed_median = np.median([p[1] for p in track])
        seed_harmonic = round(seed_median / f0)
        
        scale_factor = harmonic_n / seed_harmonic
        
        return [(t, f * scale_factor) for t, f in track]
    
    def _search_track(self, expected_track: List[Tuple[float, float]], 
                      search_bandwidth: float = 100.0) -> Optional[List[Tuple[float, float]]]:
        """Search for actual track near expected position."""
        # Use peak detection along expected track
        found_points = []
        
        for t, expected_f in expected_track:
            t_idx = np.argmin(np.abs(self.times - t))
            
            # Search window
            f_min = expected_f - search_bandwidth
            f_max = expected_f + search_bandwidth
            f_min_idx = np.argmin(np.abs(self.freqs - f_min))
            f_max_idx = np.argmin(np.abs(self.freqs - f_max))
            
            # Find peak in window
            window = self.spectrogram[f_min_idx:f_max_idx, t_idx]
            if len(window) > 0:
                peak_idx = np.argmax(window)
                actual_f = self.freqs[f_min_idx + peak_idx]
                found_points.append((t, actual_f))
        
        return found_points if len(found_points) > 3 else None
    
    def _calculate_correlation(self, expected: List, found: List) -> float:
        """Calculate shape correlation between tracks."""
        if len(expected) != len(found):
            return 0.0
        
        expected_f = np.array([p[1] for p in expected])
        found_f = np.array([p[1] for p in found])
        
        # Normalize and correlate
        expected_norm = (expected_f - expected_f.mean()) / (expected_f.std() + 1e-6)
        found_norm = (found_f - found_f.mean()) / (found_f.std() + 1e-6)
        
        return np.corrcoef(expected_norm, found_norm)[0, 1]
```

### 4.3 Implementation Steps

#### Step 4.3.1: Create HarmonicDetector Class
**File:** `audio_visualizer/core/harmonic_detector.py`

#### Step 4.3.2: Add UI Integration
- "Find Harmonics" context menu on annotation
- Dialog showing found harmonics with confidence
- Option to auto-create linked annotations
- Visual highlighting of harmonic relationships

#### Step 4.3.3: Update Event Grouping
- Annotations with same `event_id` are harmonics of same source
- Event summary includes all harmonic info

---

## 5. Priority 4: Advanced Export & Event Grouping

### 5.1 Problem Statement
Need to:
1. Group related annotations (harmonics) into "events"
2. Export images with proper axes
3. Generate comprehensive reports
4. Aggregate across multiple files

### 5.2 Event Grouping Algorithm

#### 5.2.1 Automatic Event Detection
```python
class EventGrouper:
    """Groups annotations into events based on temporal and harmonic relationships."""
    
    def group_annotations(self, annotations: List[Annotation]) -> List[Event]:
        """
        Group annotations into events.
        
        Criteria:
        1. Temporal overlap or proximity
        2. Harmonic relationship (frequency ratios)
        3. Similar slope patterns
        """
        events = []
        used = set()
        
        for ann in annotations:
            if ann.id in used:
                continue
            
            # Find all related annotations
            related = self._find_related(ann, annotations, used)
            
            if related:
                event = self._create_event(related)
                events.append(event)
                used.update(a.id for a in related)
        
        return events
    
    def _find_related(self, seed: Annotation, 
                      all_annotations: List[Annotation],
                      used: set) -> List[Annotation]:
        """Find annotations related to seed."""
        related = [seed]
        
        for ann in all_annotations:
            if ann.id in used or ann.id == seed.id:
                continue
            
            # Check temporal overlap
            if not self._temporal_overlap(seed, ann):
                continue
            
            # Check harmonic relationship
            if self._is_harmonic(seed, ann):
                related.append(ann)
        
        return related
    
    def _temporal_overlap(self, a: Annotation, b: Annotation, 
                          tolerance: float = 2.0) -> bool:
        """Check if two annotations overlap in time."""
        a_start, a_end = min(a.t_start, a.t_end), max(a.t_start, a.t_end)
        b_start, b_end = min(b.t_start, b.t_end), max(b.t_start, b.t_end)
        
        # Expand by tolerance
        a_start -= tolerance
        a_end += tolerance
        
        return not (b_end < a_start or b_start > a_end)
    
    def _is_harmonic(self, a: Annotation, b: Annotation, 
                     tolerance: float = 0.15) -> bool:
        """Check if b is a harmonic of a (or vice versa)."""
        if not a.points or not b.points:
            return False
        
        f_a = np.median([p[1] for p in a.points])
        f_b = np.median([p[1] for p in b.points])
        
        # Check integer ratios
        for n in range(1, 8):
            for m in range(1, 8):
                expected_ratio = n / m
                actual_ratio = f_b / f_a if f_a > 0 else 0
                
                if abs(actual_ratio - expected_ratio) / expected_ratio < tolerance:
                    return True
        
        return False


@dataclass
class Event:
    """Represents a grouped acoustic event."""
    event_id: str
    file_name: str
    
    # Time bounds (from all annotations)
    t_start: float
    t_end: float
    duration: float
    
    # Frequency bounds
    f_min: float
    f_max: float
    
    # Annotations in this event
    annotation_ids: List[int]
    annotation_count: int
    
    # Analysis
    estimated_f0: Optional[float]
    harmonics_detected: List[int]  # [1, 2, 3, 4] etc.
    average_snr: Optional[float]
    average_slope: Optional[float]
    
    # Metadata
    label: str = ""
    notes: str = ""
```

### 5.3 Image Export with Axes

#### 5.3.1 Implementation
```python
def export_annotation_image(annotation: Annotation,
                            spectrogram: np.ndarray,
                            times: np.ndarray,
                            freqs: np.ndarray,
                            output_path: Path,
                            include_track: bool = True,
                            dpi: int = 150) -> bool:
    """
    Export annotation as image with proper axes.
    
    Creates a matplotlib figure with:
    - Spectrogram cutout
    - Time axis (seconds)
    - Frequency axis (Hz)
    - Track overlay (if available)
    - Title with annotation info
    """
    import matplotlib.pyplot as plt
    
    # Extract region
    t_min, t_max = min(annotation.t_start, annotation.t_end), max(annotation.t_start, annotation.t_end)
    f_min, f_max = min(annotation.f_min, annotation.f_max), max(annotation.f_min, annotation.f_max)
    
    # Find indices
    t_start_idx = np.argmin(np.abs(times - t_min))
    t_end_idx = np.argmin(np.abs(times - t_max))
    f_min_idx = np.argmin(np.abs(freqs - f_min))
    f_max_idx = np.argmin(np.abs(freqs - f_max))
    
    # Extract cutout
    cutout = spectrogram[f_min_idx:f_max_idx, t_start_idx:t_end_idx]
    cutout_times = times[t_start_idx:t_end_idx]
    cutout_freqs = freqs[f_min_idx:f_max_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot spectrogram
    extent = [cutout_times[0], cutout_times[-1], cutout_freqs[0], cutout_freqs[-1]]
    im = ax.imshow(cutout, aspect='auto', origin='lower', extent=extent, cmap='magma')
    
    # Add track if available
    if include_track and annotation.points:
        track_t = [p[0] for p in annotation.points]
        track_f = [p[1] for p in annotation.points]
        ax.plot(track_t, track_f, 'c-', linewidth=2, label='Track')
        ax.plot(track_t, track_f, 'co', markersize=4)
    
    # Labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    title = f"Annotation #{annotation.id}"
    if annotation.track_label:
        title += f" - {annotation.track_label}"
    if annotation.snr_db:
        title += f" | SNR: {annotation.snr_db:.1f} dB"
    ax.set_title(title)
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return True
```

### 5.4 Unified CSV Export

#### 5.4.1 All Annotations CSV
```csv
file_name,annotation_id,t_start,t_end,duration,f_min,f_max,freq_range,snr_db,slope_hz_per_sec,harmonic_order,event_id,label,point_count
recording_001.wav,1,125.34,132.89,7.55,450,2200,1750,18.5,-175.3,2,EVT_001,vehicle,25
recording_001.wav,2,125.50,132.70,7.20,900,4400,3500,15.2,-350.6,4,EVT_001,vehicle,23
recording_002.wav,1,45.20,52.10,6.90,380,1800,1420,12.3,-145.2,1,EVT_002,unknown,20
```

#### 5.4.2 Events CSV
```csv
event_id,file_name,t_start,t_end,duration,f_min,f_max,annotation_count,harmonics,estimated_f0,avg_snr,avg_slope,label
EVT_001,recording_001.wav,125.34,132.89,7.55,450,4400,2,"2,4",462.5,16.85,-262.95,vehicle
EVT_002,recording_002.wav,45.20,52.10,6.90,380,1800,1,"1",380.0,12.30,-145.20,unknown
```

### 5.5 Implementation Steps

#### Step 5.5.1: Create EventGrouper Class
**File:** `audio_visualizer/core/event_grouper.py`

#### Step 5.5.2: Create ExportManager Class
**File:** `audio_visualizer/core/export_manager.py`

Methods:
- `export_annotation_image()`
- `export_all_annotations_csv()`
- `export_events_csv()`
- `generate_html_report()`

#### Step 5.5.3: Update UI
- Add "Export Project" menu with options
- Add "Group Events" action
- Show event grouping in annotation table

---

## 6. Priority 5: Automatic Track Detection

### 6.1 Problem Statement
Manual annotation is time-consuming. Need automatic detection that:
1. Works on entire spectrogram (general detection)
2. Works within annotation rectangle (focused detection)
3. Identifies S-curve Doppler patterns
4. Creates proper annotations with tracks

### 6.2 Reference Algorithm Analysis

The existing algorithm in `C:\Users\koren\Projects\TryCuda\simulated.py` uses:

#### 6.2.1 Pipeline
1. **Spectrogram computation** - n_fft=2048, hop=256
2. **Normalization** - Scale to 0-1
3. **Cleanup**:
   - Remove low-frequency bins (bottom 20)
   - Median subtraction per frequency bin (removes horizontal noise)
4. **Ridge detection** - Meijering filter (sigmas 1-3)
5. **Binarization** - 99.5 percentile threshold
6. **Clustering** - DBSCAN with StandardScaler
7. **Selection** - Score by aspect ratio × log(size)

#### 6.2.2 Strengths
- **Median subtraction** effectively removes constant noise per frequency
- **Meijering filter** is good for detecting line structures
- **DBSCAN** doesn't require knowing cluster count
- **Aspect ratio scoring** favors vertical (Doppler-like) structures

#### 6.2.3 Weaknesses & Improvements Needed

| Issue | Problem | Solution |
|-------|---------|----------|
| Fixed threshold (99.5%) | Misses weak signals | Adaptive threshold based on local statistics |
| Single cluster output | Misses multiple events | Return top N clusters |
| No temporal tracking | Doesn't follow S-curve | Add temporal continuity constraint |
| Fixed parameters | Not universal | Auto-tune based on spectrogram statistics |
| No shape prior | Doesn't know Doppler shape | Add S-curve template matching |
| No harmonic awareness | Treats harmonics as separate | Group by frequency ratio |

### 6.3 Improved Detection Algorithm

```python
class AutomaticTrackDetector:
    """Automatic Doppler track detection with improvements."""
    
    def __init__(self):
        self.meijering_sigmas = range(1, 5)
        self.dbscan_eps = 0.15
        self.dbscan_min_samples = 10
        self.min_track_height = 50  # bins
        self.max_clusters = 10
    
    def detect(self, spectrogram: np.ndarray, 
               times: np.ndarray, 
               freqs: np.ndarray,
               region: Optional[Tuple] = None) -> List[DetectedTrack]:
        """
        Detect Doppler tracks in spectrogram.
        
        Args:
            spectrogram: 2D array (freq x time) in dB
            times: Time axis
            freqs: Frequency axis
            region: Optional (t_min, t_max, f_min, f_max) to limit search
        
        Returns:
            List of DetectedTrack objects
        """
        # Extract region if specified
        if region:
            spec, t, f = self._extract_region(spectrogram, times, freqs, region)
        else:
            spec, t, f = spectrogram, times, freqs
        
        # Preprocessing
        spec_clean = self._preprocess(spec)
        
        # Ridge detection
        spec_ridge = self._detect_ridges(spec_clean)
        
        # Adaptive thresholding
        binary = self._adaptive_threshold(spec_ridge)
        
        # Clustering
        clusters = self._cluster_points(binary)
        
        # Score and filter clusters
        tracks = self._score_clusters(clusters, spec, t, f)
        
        # Convert to world coordinates
        tracks = self._to_world_coords(tracks, t, f, region)
        
        return tracks
    
    def _preprocess(self, spec: np.ndarray) -> np.ndarray:
        """Clean spectrogram."""
        # Normalize
        spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        
        # Remove low frequency noise (adaptive)
        low_freq_bins = max(10, int(spec.shape[0] * 0.02))
        spec_norm[:low_freq_bins, :] = 0
        
        # Median subtraction (per frequency bin)
        median = np.median(spec_norm, axis=1, keepdims=True)
        spec_clean = np.maximum(spec_norm - median, 0)
        
        return spec_clean
    
    def _detect_ridges(self, spec: np.ndarray) -> np.ndarray:
        """Apply ridge detection filter."""
        from skimage.filters import meijering
        
        ridge = meijering(spec, sigmas=self.meijering_sigmas, black_ridges=False)
        ridge = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-6)
        
        return ridge
    
    def _adaptive_threshold(self, spec: np.ndarray) -> np.ndarray:
        """Adaptive thresholding based on local statistics."""
        # Global threshold
        global_thresh = np.percentile(spec, 98)
        
        # Local threshold (for weak signals)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(spec, size=50)
        local_std = np.sqrt(uniform_filter(spec**2, size=50) - local_mean**2)
        local_thresh = local_mean + 2 * local_std
        
        # Combine: pixel must exceed both thresholds
        binary = (spec > global_thresh) | (spec > local_thresh)
        
        return binary
    
    def _cluster_points(self, binary: np.ndarray) -> List[np.ndarray]:
        """Cluster binary mask points."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        points = np.column_stack(np.where(binary))
        
        if len(points) < self.dbscan_min_samples:
            return []
        
        # Normalize coordinates
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        
        # DBSCAN
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = db.fit_predict(points_scaled)
        
        # Extract clusters
        clusters = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            cluster_points = points[labels == label]
            clusters.append(cluster_points)
        
        return clusters
    
    def _score_clusters(self, clusters: List[np.ndarray], 
                        spec: np.ndarray,
                        times: np.ndarray,
                        freqs: np.ndarray) -> List[DetectedTrack]:
        """Score and rank clusters."""
        scored = []
        
        for cluster in clusters:
            if len(cluster) < 10:
                continue
            
            # Bounding box
            f_min, t_min = cluster.min(axis=0)
            f_max, t_max = cluster.max(axis=0)
            
            height = f_max - f_min
            width = t_max - t_min + 1
            
            if height < self.min_track_height:
                continue
            
            # Aspect ratio (prefer vertical)
            aspect = height / width
            
            # Size
            size = len(cluster)
            
            # Intensity (average spectrogram value at cluster points)
            intensity = np.mean(spec[cluster[:, 0], cluster[:, 1]])
            
            # Continuity (how connected is the cluster)
            continuity = self._measure_continuity(cluster)
            
            # Combined score
            score = aspect * np.log(size) * intensity * continuity
            
            # Extract track centerline
            track_points = self._extract_centerline(cluster, times, freqs)
            
            scored.append(DetectedTrack(
                points=track_points,
                score=score,
                bounds=(t_min, t_max, f_min, f_max),
                cluster_size=size
            ))
        
        # Sort by score and return top N
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:self.max_clusters]
    
    def _measure_continuity(self, cluster: np.ndarray) -> float:
        """Measure how continuous/connected a cluster is."""
        # Simple: ratio of actual points to bounding box area
        f_min, t_min = cluster.min(axis=0)
        f_max, t_max = cluster.max(axis=0)
        
        bbox_area = (f_max - f_min + 1) * (t_max - t_min + 1)
        fill_ratio = len(cluster) / bbox_area
        
        return fill_ratio
    
    def _extract_centerline(self, cluster: np.ndarray,
                            times: np.ndarray,
                            freqs: np.ndarray) -> List[Tuple[float, float]]:
        """Extract track centerline from cluster."""
        # Group by time column
        time_groups = {}
        for f_idx, t_idx in cluster:
            if t_idx not in time_groups:
                time_groups[t_idx] = []
            time_groups[t_idx].append(f_idx)
        
        # For each time, take median frequency
        centerline = []
        for t_idx in sorted(time_groups.keys()):
            f_idx = int(np.median(time_groups[t_idx]))
            t = times[t_idx] if t_idx < len(times) else times[-1]
            f = freqs[f_idx] if f_idx < len(freqs) else freqs[-1]
            centerline.append((t, f))
        
        return centerline


@dataclass
class DetectedTrack:
    """A detected track from automatic detection."""
    points: List[Tuple[float, float]]
    score: float
    bounds: Tuple[int, int, int, int]  # t_min, t_max, f_min, f_max (indices)
    cluster_size: int
```

### 6.4 Implementation Steps

#### Step 6.4.1: Create AutomaticTrackDetector Class
**File:** `audio_visualizer/core/auto_detector.py`

#### Step 6.4.2: Add UI Integration
- "Auto Detect Tracks" menu option
- "Detect in Region" context menu on annotation
- Dialog to review and accept/reject detections
- Batch detection across multiple files

#### Step 6.4.3: Create Detected Annotations
- Convert DetectedTrack to Annotation
- Auto-create rectangle bounds
- Set track points
- Run analysis (SNR, slope)

---

## 7. Priority 6: GPS Integration (Future)

### 7.1 Overview
Future integration for:
- Sensor location from GPS
- Source location estimation
- Multi-sensor fusion
- Map visualization

### 7.2 Data Requirements
```json
{
  "sensor_gps": {
    "latitude": 32.0853,
    "longitude": 34.7818,
    "altitude": 50.0,
    "timestamp": "2025-06-02T10:30:00Z"
  },
  "source_gps": {
    "track": [
      {"lat": 32.0860, "lon": 34.7820, "time": 125.34},
      {"lat": 32.0855, "lon": 34.7815, "time": 130.00}
    ]
  }
}
```

### 7.3 Planned Features
- Load GPS tracks
- Calculate true velocity from GPS
- Compare Doppler-estimated vs GPS velocity
- Visualize on map

---

## 8. Priority 7: Doppler Model Limitations

### 8.1 Current Model Issues

The current Doppler model in `doppler_analysis.py` has significant limitations:

#### 8.1.1 Missing Information
| Required | Available | Impact |
|----------|-----------|--------|
| Sensor position | ❌ | Cannot calculate true CPA distance |
| Source trajectory | ❌ | Cannot validate velocity |
| Speed of sound | Assumed 343 m/s | Varies with temperature |
| Harmonic order | ❌ | f0 estimation unreliable |

#### 8.1.2 Model Assumptions
- Source moves in straight line (may not be true)
- Constant velocity (acceleration not modeled)
- Single CPA point (complex trajectories not supported)
- Stationary sensor (moving sensor not supported)

#### 8.1.3 Fit Reliability
Without ground truth, the fit can converge to multiple solutions:
- Different combinations of (v, x_cpa, f0) can produce similar curves
- The problem is under-constrained

### 8.2 Recommended Actions

#### 8.2.1 Short Term (Current)
- **Remove velocity and CPA distance from annotation table** - They are unreliable
- **Keep slope calculation** - This is directly measurable
- **Keep SNR calculation** - This is directly measurable
- **Store curve points** - For future analysis with GPS

#### 8.2.2 Medium Term (With GPS)
- Add GPS data loading
- Calculate true source-sensor geometry
- Constrained Doppler fit with known positions
- Validate model against GPS-derived velocity

#### 8.2.3 Long Term
- Multi-sensor fusion
- Real-time tracking
- Machine learning for source classification

### 8.3 UI Changes Required

Remove from annotation table:
- ~~Speed (km/h)~~ column
- ~~CPA Dist (m)~~ column

Keep/Add:
- SNR (dB) column
- Slope (Hz/s) column
- Event ID column
- Harmonic Order column

---

## 9. Reference: Existing Detection Algorithm Analysis

### 9.1 Source Code Location
`C:\Users\koren\Projects\TryCuda\simulated.py`

### 9.2 Algorithm Summary

```
Input: Audio file
   ↓
1. Load audio (librosa)
   ↓
2. Compute spectrogram (n_fft=2048, hop=256)
   ↓
3. Convert to dB, normalize to [0,1]
   ↓
4. Remove bottom 20 frequency bins
   ↓
5. Median subtraction per frequency bin
   ↓
6. Meijering ridge filter (sigmas 1-3)
   ↓
7. Normalize ridge output to [0,1]
   ↓
8. Binarize at 99.5 percentile
   ↓
9. Extract point coordinates
   ↓
10. StandardScaler normalization
   ↓
11. DBSCAN clustering (eps=0.15, min_samples=10)
   ↓
12. Score clusters: aspect_ratio × log(size)
   ↓
13. Select best cluster (height > 50)
   ↓
Output: Binary mask of selected cluster
```

### 9.3 Key Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_fft | 2048 | Frequency resolution |
| hop_length | 256 | Time resolution |
| Low freq cutoff | 20 bins | Remove DC/low noise |
| Meijering sigmas | 1-3 | Ridge width detection |
| Threshold percentile | 99.5 | Binary mask creation |
| DBSCAN eps | 0.15 | Neighbor distance |
| DBSCAN min_samples | 10 | Minimum cluster density |
| Min height | 50 bins | Filter small clusters |

### 9.4 Improvements for Integration
1. Return multiple clusters, not just best
2. Add adaptive thresholding
3. Extract centerline track points
4. Calculate SNR for each cluster
5. Group harmonics by frequency ratio
6. Create proper Annotation objects

---

## 10. Implementation Checklist

### Phase 1: Data Management (Priority 1)
- [ ] Create `ProjectManager` class
- [ ] Define JSON schemas for project, file metadata, annotations
- [ ] Implement project create/load/save
- [ ] Update `AnnotationManager` to save spectrogram params
- [ ] Add Project menu to UI
- [ ] Test multi-file project workflow

### Phase 2: Track Analysis (Priority 2)
- [ ] Create `TrackAnalyzer` class
- [ ] Implement adaptive bandwidth SNR calculation
- [ ] Implement slope analysis
- [ ] Add analysis fields to `Annotation` dataclass
- [ ] Update annotation table with SNR, Slope columns
- [ ] Remove Speed, CPA Distance columns from table
- [ ] Add "Analyze Track" UI action
- [ ] Test on real data

### Phase 3: Harmonic Detection (Priority 3)
- [ ] Create `HarmonicDetector` class
- [ ] Implement f0 estimation
- [ ] Implement harmonic search
- [ ] Implement correlation matching
- [ ] Add "Find Harmonics" UI action
- [ ] Update annotation linking (event_id, harmonic_order)
- [ ] Test harmonic grouping

### Phase 4: Advanced Export (Priority 4)
- [ ] Create `EventGrouper` class
- [ ] Implement temporal/harmonic grouping
- [ ] Create `ExportManager` class
- [ ] Implement image export with axes
- [ ] Implement unified CSV export
- [ ] Implement events CSV export
- [ ] Add export UI options
- [ ] Generate sample reports

### Phase 5: Automatic Detection (Priority 5)
- [ ] Create `AutomaticTrackDetector` class
- [ ] Integrate Meijering filter
- [ ] Implement improved DBSCAN pipeline
- [ ] Implement centerline extraction
- [ ] Add "Auto Detect" UI action
- [ ] Add "Detect in Region" context menu
- [ ] Test detection accuracy
- [ ] Tune parameters

### Phase 6: GPS Integration (Future)
- [ ] Define GPS data schema
- [ ] Create GPS loader
- [ ] Implement geometry calculations
- [ ] Update Doppler model with constraints
- [ ] Add map visualization

---

## Document History
- **2025-06-02**: Initial creation based on analysis session
- **Version**: 1.0
- **Author**: Development session with user requirements

---

## Quick Reference: File Locations

| Component | File |
|-----------|------|
| Main window | `audio_visualizer/ui/main_window.py` |
| Annotation data | `audio_visualizer/ui/annotation_data.py` |
| Annotation manager | `audio_visualizer/ui/annotation_manager.py` |
| Annotation renderer | `audio_visualizer/ui/annotation_renderer.py` |
| Annotation table | `audio_visualizer/ui/annotation_table.py` |
| Spectrogram canvas | `audio_visualizer/ui/vispy_canvas.py` |
| Doppler analysis | `audio_visualizer/core/doppler_analysis.py` |
| Spectrogram engine | `audio_visualizer/engines/spectrogram_engine.py` |
| Reference detector | `C:\Users\koren\Projects\TryCuda\simulated.py` |

