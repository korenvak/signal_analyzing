# Event Tagging System - Implementation Plan

## Overview

A new **Event Tagging Mode** for marking acoustic events on spectrograms. Unlike the existing annotation system (which draws rectangles), this mode:

1. Uses **two vertical lines** to define event start/end times
2. Collects metadata: harmonic number, SNR estimation
3. **Auto-captures images** with axes (matplotlib)
4. **Parses filenames** like `pixel - 1008 - 2025-29-10 11-04-47 - 2025-29-10 12-20-49` to compute absolute timestamps
5. Stores events in a **separate CSV** (not mixed with rectangle annotations)
6. Supports **multi-file sessions** (lines cleared on file switch, CSV continues)
7. GUI for **editing/deleting events** from the table
8. Supports **continuing existing CSV** (append mode)

---

## 1. Data Model

### 1.1 Event Data Class

**File**: `audio_visualizer/ui/event_data.py` (new file)

```python
@dataclass
class TaggedEvent:
    """Represents a tagged acoustic event between two time markers."""
    id: int                              # Unique event ID

    # File info
    audio_file: str                      # Full path to audio file
    sensor_name: str                     # e.g., "pixel" (parsed from filename)
    sensor_id: int                       # e.g., 1008 (parsed from filename)

    # Time info (relative to audio file)
    t_start_relative: float              # Start time in seconds (relative to file start)
    t_end_relative: float                # End time in seconds (relative to file start)

    # Time info (absolute - parsed from filename)
    file_start_time: Optional[datetime]  # File start time from filename
    file_end_time: Optional[datetime]    # File end time from filename
    event_start_absolute: Optional[datetime]  # Computed absolute start
    event_end_absolute: Optional[datetime]    # Computed absolute end

    # User-provided metadata
    harmonic_number: Optional[int] = None     # User input
    snr_estimate_db: Optional[float] = None   # User input
    notes: str = ""                           # Optional notes

    # Auto-generated
    image_path: str = ""                 # Path to saved screenshot
    created_at: datetime                 # When event was created

    def to_csv_row(self) -> dict: ...

    @classmethod
    def from_csv_row(cls, row: dict) -> 'TaggedEvent': ...
```

### 1.2 Filename Parser

**File**: `audio_visualizer/core/filename_parser.py` (new file)

```python
def parse_pixel_filename(filename: str) -> Optional[dict]:
    """
    Parse filename format: 'pixel - ID - START_TIME - END_TIME'
    Example: 'pixel - 1008 - 2025-29-10 11-04-47 - 2025-29-10 12-20-49'

    Returns:
        {
            'sensor_name': 'pixel',
            'sensor_id': 1008,
            'start_time': datetime(2025, 10, 29, 11, 4, 47),
            'end_time': datetime(2025, 10, 29, 12, 20, 49)
        }
    or None if filename doesn't match expected format.

    Note: Date format appears to be yyyy-dd-mm HH-MM-SS
    """
```

### 1.3 CSV Format

**Columns**:
```
id, audio_file, sensor_name, sensor_id,
t_start_relative, t_end_relative,
event_start_absolute, event_end_absolute,
harmonic_number, snr_estimate_db, notes,
image_path, created_at
```

**Example**:
```csv
id,audio_file,sensor_name,sensor_id,t_start_relative,t_end_relative,event_start_absolute,event_end_absolute,harmonic_number,snr_estimate_db,notes,image_path,created_at
1,pixel - 1008 - 2025-29-10 11-04-47.flac,pixel,1008,125.500,130.250,2025-10-29 11:06:52.500000,2025-10-29 11:07:17.250000,3,15.5,,events/event_001.png,2025-12-08 14:30:00
```

---

## 2. Event Manager

**File**: `audio_visualizer/ui/event_manager.py` (new file)

```python
class EventManager:
    """Manages tagged events, CSV persistence, and image capture."""

    def __init__(self):
        self.events: List[TaggedEvent] = []
        self.csv_path: Optional[Path] = None
        self.image_dir: Optional[Path] = None
        self.next_id: int = 1
        self.filename_parser = ...

    # CSV Operations
    def create_new_session(self, csv_path: Path, image_dir: Path): ...
    def load_existing_csv(self, csv_path: Path): ...  # Continue existing
    def save_to_csv(self): ...                         # Full save
    def append_event_to_csv(self, event: TaggedEvent): ...  # Incremental

    # Event Operations
    def add_event(self, audio_file: str, t_start: float, t_end: float,
                  harmonic: Optional[int], snr: Optional[float]) -> TaggedEvent: ...
    def delete_event(self, event_id: int): ...
    def update_event(self, event_id: int, **kwargs): ...
    def get_events_for_file(self, audio_file: str) -> List[TaggedEvent]: ...

    # Image Capture
    def capture_event_image(self, event: TaggedEvent,
                           spectrogram_data: np.ndarray,
                           times: np.ndarray, freqs: np.ndarray,
                           colormap: str = 'plasma') -> str: ...

    # SNR Estimation
    def estimate_snr(self, S: np.ndarray, times: np.ndarray, freqs: np.ndarray,
                    t_start: float, t_end: float,
                    f_min: float, f_max: float) -> float:
        """Estimate SNR in dB for the given region.

        Method: SNR = max_value - median_value (in dB)
        This gives a rough estimate of signal strength above noise floor.
        """
        ...
```

### 2.1 Image Capture with Axes

```python
def capture_event_image(self, event, S, times, freqs, colormap='plasma'):
    """Save spectrogram region as image with proper axes using matplotlib."""
    import matplotlib.pyplot as plt

    # Extract region
    t_mask = (times >= event.t_start_relative) & (times <= event.t_end_relative)
    region = S[:, t_mask]
    region_times = times[t_mask]

    # Create figure with axes
    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [region_times[0], region_times[-1], freqs[0], freqs[-1]]
    im = ax.imshow(region, aspect='auto', origin='lower',
                   extent=extent, cmap=colormap)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Event {event.id} - {event.sensor_name} {event.sensor_id}')

    plt.colorbar(im, ax=ax, label='dB')

    # Save
    image_path = self.image_dir / f"event_{event.id:04d}.png"
    fig.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(image_path)
```

---

## 3. GUI Components

### 3.1 Event Mode in Canvas

**File**: `audio_visualizer/ui/vispy_canvas.py` (modify)

Add new mode alongside existing modes:
- `annotation_mode` (existing)
- `measurement_mode` (existing)
- `curve_mode` (existing)
- **`event_mode`** (new)

```python
# New state variables
self.event_mode = False
self.event_line_1: Optional[Line] = None  # First vertical line
self.event_line_2: Optional[Line] = None  # Second vertical line
self.event_t1: Optional[float] = None     # First time marker
self.event_t2: Optional[float] = None     # Second time marker
self._on_event_created_callback = None    # Callback(t1, t2)

def set_event_mode(self, enabled: bool):
    """Enable/disable event tagging mode."""
    self.event_mode = enabled
    if enabled:
        # Disable other modes
        self.annotation_mode = False
        self.measurement_mode = False
        self.curve_mode = False
    else:
        # Clear event markers
        self.clear_event_markers()
    logger.info(f"Event mode: {'ON' if enabled else 'OFF'}")

def add_event_marker(self, time: float):
    """Add vertical line at given time position."""
    if self.event_t1 is None:
        # First click - create line 1
        self.event_t1 = time
        self.event_line_1 = self._create_vertical_line(time, color='lime')
    elif self.event_t2 is None:
        # Second click - create line 2 and trigger callback
        self.event_t2 = time
        self.event_line_2 = self._create_vertical_line(time, color='lime')

        # Trigger callback with ordered times
        t_start = min(self.event_t1, self.event_t2)
        t_end = max(self.event_t1, self.event_t2)
        if self._on_event_created_callback:
            self._on_event_created_callback(t_start, t_end)
```

### 3.2 Event Input Dialog

**File**: `audio_visualizer/ui/event_dialog.py` (new file)

```python
class EventInputDialog(QDialog):
    """Dialog for entering event metadata after marking time region."""

    def __init__(self, t_start: float, t_end: float,
                 absolute_start: Optional[datetime] = None,
                 absolute_end: Optional[datetime] = None,
                 suggested_snr: Optional[float] = None,  # Auto-calculated
                 current_f_min: float = 0,               # From current view
                 current_f_max: float = 22050,           # From current view
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tag Event")

        # Show time info (read-only)
        QLabel(f"Relative Time: {t_start:.3f}s - {t_end:.3f}s")
        if absolute_start:
            QLabel(f"Absolute Time: {absolute_start} - {absolute_end}")

        # Frequency range (editable, defaults to current view)
        self.f_min_spin = QDoubleSpinBox()  # Default: current_f_min
        self.f_max_spin = QDoubleSpinBox()  # Default: current_f_max

        # Input fields
        self.harmonic_spin = QSpinBox()  # 1-20 range
        self.snr_spin = QDoubleSpinBox() # -20 to +60 dB
        if suggested_snr:
            self.snr_spin.setValue(suggested_snr)  # Pre-fill with suggestion
        self.notes_edit = QLineEdit()

        # Buttons
        self.save_btn = QPushButton("Save Event")
        self.cancel_btn = QPushButton("Cancel")
```

### 3.3 Event Table Widget

**File**: `audio_visualizer/ui/event_table.py` (new file)

```python
class EventTableWidget(QTableWidget):
    """Table for displaying and editing tagged events."""

    event_selected = Signal(int)   # event_id
    event_deleted = Signal(int)    # event_id
    event_updated = Signal(int)    # event_id

    COLUMNS = [
        'ID', 'File', 'Sensor', 'ID#',
        't_start', 't_end',
        'Absolute Start', 'Absolute End',
        'Harmonic', 'SNR (dB)', 'Notes', 'Image'
    ]

    def add_event(self, event: TaggedEvent) -> int: ...
    def update_event(self, event: TaggedEvent): ...
    def remove_event(self, event_id: int): ...
    def clear_all(self): ...

    # Context menu: Delete, Edit, View Image, Go to Event
    def show_context_menu(self, pos): ...
```

### 3.4 Event Panel Widget

**File**: `audio_visualizer/ui/event_panel.py` (new file)

```python
class EventPanel(QWidget):
    """Panel containing event controls and table."""

    def __init__(self, parent=None):
        # Mode toggle button
        self.mode_btn = QPushButton("Event Mode (E)")

        # Session controls
        self.new_session_btn = QPushButton("New Session...")
        self.load_session_btn = QPushButton("Continue Session...")

        # Current session info
        self.session_label = QLabel("No session active")

        # Event table
        self.event_table = EventTableWidget()

        # Status
        self.event_count_label = QLabel("Events: 0")
```

---

## 4. Integration with Main Window

**File**: `audio_visualizer/ui/main_window.py` (modify)

### 4.1 Setup

```python
def setup_ui(self):
    ...
    # Add Event Manager
    self.event_manager = EventManager()

    # Add Event Panel (as tab or collapsible section)
    self.event_panel = EventPanel()

    # Connect signals
    self.event_panel.mode_btn.clicked.connect(self.toggle_event_mode)
    self.spectrogram_canvas.set_event_created_callback(self.on_event_region_marked)
```

### 4.2 Event Mode Toggle

```python
def toggle_event_mode(self):
    """Toggle event tagging mode."""
    enabled = not self.spectrogram_canvas.event_mode
    self.spectrogram_canvas.set_event_mode(enabled)
    self.event_panel.mode_btn.setChecked(enabled)

    # Update status
    if enabled:
        self.status_widget.show_message("Event Mode: Click to mark start/end times")
```

### 4.3 Event Creation Flow

```python
def on_event_region_marked(self, t_start: float, t_end: float):
    """Called when user marks two time points in event mode."""

    # Check if session is active
    if not self.event_manager.csv_path:
        QMessageBox.warning(self, "No Session",
            "Please create or load an event session first.")
        return

    # Parse filename for absolute times
    parsed = parse_pixel_filename(self.current_file)
    absolute_start = absolute_end = None
    if parsed:
        file_start = parsed['start_time']
        absolute_start = file_start + timedelta(seconds=t_start)
        absolute_end = file_start + timedelta(seconds=t_end)

    # Show input dialog
    dialog = EventInputDialog(t_start, t_end, absolute_start, absolute_end, self)
    if dialog.exec() == QDialog.Accepted:
        # Create event
        event = self.event_manager.add_event(
            audio_file=self.current_file,
            t_start=t_start,
            t_end=t_end,
            harmonic=dialog.harmonic_spin.value() or None,
            snr=dialog.snr_spin.value() or None
        )

        # Capture image
        S, times, freqs = self._get_current_spectrogram_data()
        self.event_manager.capture_event_image(event, S, times, freqs)

        # Save to CSV
        self.event_manager.append_event_to_csv(event)

        # Update table
        self.event_panel.event_table.add_event(event)

        # Clear markers for next event
        self.spectrogram_canvas.clear_event_markers()
```

### 4.4 File Switch Handling

```python
def load_audio_file(self, file_path: str):
    """Load audio file."""
    ...
    # Clear event markers when switching files (but keep CSV data)
    self.spectrogram_canvas.clear_event_markers()
    ...
```

---

## 5. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **E** | Toggle Event Mode |
| **Escape** | Clear current event markers (in event mode) |

---

## 6. File Structure Summary

### New Files:
```
audio_visualizer/
├── core/
│   └── filename_parser.py          # Filename parsing logic
├── ui/
│   ├── event_data.py               # TaggedEvent dataclass
│   ├── event_manager.py            # Event management + CSV + images
│   ├── event_dialog.py             # Event input dialog
│   ├── event_table.py              # Event table widget
│   └── event_panel.py              # Event panel container
```

### Modified Files:
```
audio_visualizer/ui/
├── vispy_canvas.py                 # Add event_mode, vertical lines
├── main_window.py                  # Integration, event handling
└── status_widget.py                # Event mode status display
```

---

## 7. Implementation Order

1. **Phase 1: Core Data Model**
   - [ ] `filename_parser.py` - Parse pixel filenames
   - [ ] `event_data.py` - TaggedEvent dataclass

2. **Phase 2: Event Manager**
   - [ ] `event_manager.py` - CSV handling, image capture

3. **Phase 3: Canvas Support**
   - [ ] Modify `vispy_canvas.py` - Add event mode, vertical lines

4. **Phase 4: GUI Components**
   - [ ] `event_dialog.py` - Input dialog
   - [ ] `event_table.py` - Event table widget
   - [ ] `event_panel.py` - Panel container

5. **Phase 5: Integration**
   - [ ] Modify `main_window.py` - Wire everything together
   - [ ] Modify `status_widget.py` - Event mode indicator

6. **Phase 6: Testing**
   - [ ] Unit tests for filename parser
   - [ ] Integration tests for event flow
   - [ ] Manual testing with real files

---

## 8. Design Decisions (Confirmed)

1. **SNR Estimation**: **Both** - Calculate suggested SNR from spectrogram data (max vs median in region), but allow user to override in dialog

2. **Frequency Range**: **Both** - Default to current view's frequency range, but allow user to adjust f_min/f_max in dialog

3. **Image Content**: Event region **with padding** (~10% time context before/after) for better visualization

4. **Date Format**: `yyyy-dd-mm HH-MM-SS` (day before month, as in filename examples)

5. **Multiple Events Per File**: Yes - user can mark multiple events per file

6. **Session Directory Structure**: Approved
   ```
   session_folder/
   ├── events.csv
   └── images/
       ├── event_0001.png
       ├── event_0002.png
       └── ...
   ```
