"""
Annotation data model for rectangle-based annotations with harmonic linking.
"""
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple
from pathlib import Path


@dataclass
class Annotation:
    """Represents a single annotation rectangle on the spectrogram.
    
    Supports harmonic linking: annotations can be linked to parent annotations
    and have associated ridge (track) data.
    """
    id: int
    file_name: str
    t_start: float  # Start time in seconds
    t_end: float    # End time in seconds
    f_min: float    # Lower frequency in Hz
    f_max: float    # Upper frequency in Hz
    
    # Harmonic linking fields
    harmonic_index: Optional[int] = None      # Deprecated, use harmonic_order
    harmonic_order: Optional[int] = None      # k=1 for fundamental, k=2,3,4 for harmonics, negative for subharmonics
    parent_rect_id: Optional[int] = None      # ID of the seed annotation that generated this one
    event_id: Optional[int] = None            # Groups related annotations (same physical event)
    
    # Quality metrics (from track analysis)
    snr_db: Optional[float] = None            # Signal-to-noise ratio in dB
    slope_hz_per_sec: Optional[float] = None  # Track slope in Hz/second
    snr_estimate: Optional[float] = None      # Legacy field
    ridge_quality: Optional[float] = None     # Quality score from ridge extraction
    harmonic_similarity: Optional[float] = None  # Similarity score to parent
    
    # Analysis parameters (for reproducibility)
    analysis_params: Optional[Dict] = None    # FFT size, overlap, window, etc.
    
    # Ridge data (stored separately but referenced here)
    ridge_data: Optional[Dict] = None         # Ridge dict if extracted
    
    # User labels
    track_label: str = ""
    is_approved: bool = False
    
    # Visibility controls
    is_visible: bool = True  # Whether annotation rectangle is visible
    show_doppler_curve: bool = True  # Whether to show the Doppler curve for this annotation
    
    # Doppler Analysis Data
    points: List[Tuple[float, float]] = field(default_factory=list)  # User drawn points [(t, f), ...]
    doppler_result: Optional[Dict] = None  # Dictionary representation of DopplerResult
    
    # Graphics reference (not serialized)
    graphics_handle: Any = None  # Reference to the rectangle visual in VisPy
    doppler_curve_handle: Any = None  # Reference to the Doppler curve visual in VisPy
    
    def to_dict(self) -> dict:
        """Convert annotation to dictionary (for JSON serialization)."""
        d = {
            'id': self.id,
            'file_name': self.file_name,
            't_start': self.t_start,
            't_end': self.t_end,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'harmonic_index': self.harmonic_index,
            'harmonic_order': self.harmonic_order,
            'parent_rect_id': self.parent_rect_id,
            'event_id': self.event_id,
            'snr_db': self.snr_db,
            'slope_hz_per_sec': self.slope_hz_per_sec,
            'snr_estimate': self.snr_estimate,
            'ridge_quality': self.ridge_quality,
            'harmonic_similarity': self.harmonic_similarity,
            'analysis_params': self.analysis_params,
            'track_label': self.track_label,
            'is_approved': self.is_approved,
            'is_visible': self.is_visible,
            'show_doppler_curve': self.show_doppler_curve,
            'points': self.points,
            'doppler_result': self.doppler_result
        }
        # Include ridge data if present
        if self.ridge_data is not None:
            d['ridge_data'] = self.ridge_data
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Annotation':
        """Create annotation from dictionary (for JSON deserialization)."""
        ann = cls(
            id=data['id'],
            file_name=data.get('file_name', ''),
            t_start=data['t_start'],
            t_end=data['t_end'],
            f_min=data['f_min'],
            f_max=data['f_max'],
            harmonic_index=data.get('harmonic_index'),
            harmonic_order=data.get('harmonic_order'),
            parent_rect_id=data.get('parent_rect_id'),
            event_id=data.get('event_id'),
            snr_db=data.get('snr_db'),
            slope_hz_per_sec=data.get('slope_hz_per_sec'),
            snr_estimate=data.get('snr_estimate'),
            ridge_quality=data.get('ridge_quality'),
            harmonic_similarity=data.get('harmonic_similarity'),
            analysis_params=data.get('analysis_params'),
            ridge_data=data.get('ridge_data'),
            track_label=data.get('track_label', ''),
            is_approved=data.get('is_approved', False),
            is_visible=data.get('is_visible', True),
            show_doppler_curve=data.get('show_doppler_curve', True),
            graphics_handle=None  # Will be set when recreating visuals
        )
        
        # Set mutable fields after creation
        ann.points = data.get('points', [])
        ann.doppler_result = data.get('doppler_result')
        return ann
    
    @property
    def width(self) -> float:
        """Width of the annotation in time (seconds)."""
        return abs(self.t_end - self.t_start)
    
    @property
    def height(self) -> float:
        """Height of the annotation in frequency (Hz)."""
        return abs(self.f_max - self.f_min)
    
    def contains_point(self, time: float, freq: float) -> bool:
        """Check if a point (time, freq) is inside this annotation."""
        t_min, t_max = min(self.t_start, self.t_end), max(self.t_start, self.t_end)
        f_min, f_max = min(self.f_min, self.f_max), max(self.f_min, self.f_max)
        return (t_min <= time <= t_max) and (f_min <= freq <= f_max)

