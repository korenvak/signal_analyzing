"""
Event Grouper
Groups annotations into events based on temporal and harmonic relationships.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


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
    estimated_f0: Optional[float] = None
    harmonics_detected: List[int] = None  # [1, 2, 3, 4] etc.
    average_snr: Optional[float] = None
    average_slope: Optional[float] = None
    
    # Metadata
    label: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if self.harmonics_detected is None:
            self.harmonics_detected = []


class EventGrouper:
    """Groups annotations into events based on temporal and harmonic relationships."""
    
    def __init__(self, temporal_tolerance: float = 2.0, harmonic_tolerance: float = 0.15):
        """
        Args:
            temporal_tolerance: Time overlap tolerance in seconds
            harmonic_tolerance: Frequency ratio tolerance for harmonic detection
        """
        self.temporal_tolerance = temporal_tolerance
        self.harmonic_tolerance = harmonic_tolerance
    
    def group_annotations(self, annotations: List[Dict]) -> List[Event]:
        """
        Group annotations into events.
        
        Args:
            annotations: List of annotation dictionaries
        
        Returns:
            List of Event objects
        """
        if not annotations:
            return []
        
        events = []
        used = set()
        event_counter = 1
        
        for ann in annotations:
            if ann.get('id') in used:
                continue
            
            # Find all related annotations
            related = self._find_related(ann, annotations, used)
            
            if related:
                event = self._create_event(related, event_counter, ann.get('file_name', ''))
                events.append(event)
                used.update(a.get('id') for a in related)
                event_counter += 1
        
        # Process remaining ungrouped annotations as single-event
        for ann in annotations:
            if ann.get('id') not in used:
                event = self._create_event([ann], event_counter, ann.get('file_name', ''))
                events.append(event)
                event_counter += 1
        
        return events
    
    def _find_related(self, seed: Dict, all_annotations: List[Dict], used: set) -> List[Dict]:
        """Find annotations related to seed."""
        related = [seed]
        
        for ann in all_annotations:
            if ann.get('id') in used or ann.get('id') == seed.get('id'):
                continue
            
            # Check temporal overlap
            if not self._temporal_overlap(seed, ann):
                continue
            
            # Check harmonic relationship
            if self._is_harmonic(seed, ann):
                related.append(ann)
        
        return related
    
    def _temporal_overlap(self, a: Dict, b: Dict, tolerance: Optional[float] = None) -> bool:
        """Check if two annotations overlap in time."""
        if tolerance is None:
            tolerance = self.temporal_tolerance
        
        a_start = min(a.get('t_start', 0), a.get('t_end', 0))
        a_end = max(a.get('t_start', 0), a.get('t_end', 0))
        b_start = min(b.get('t_start', 0), b.get('t_end', 0))
        b_end = max(b.get('t_start', 0), b.get('t_end', 0))
        
        # Expand by tolerance
        a_start -= tolerance
        a_end += tolerance
        
        return not (b_end < a_start or b_start > a_end)
    
    def _is_harmonic(self, a: Dict, b: Dict, tolerance: Optional[float] = None) -> bool:
        """Check if b is a harmonic of a (or vice versa)."""
        if tolerance is None:
            tolerance = self.harmonic_tolerance
        
        # Get frequency from points or bounds
        f_a = self._get_median_frequency(a)
        f_b = self._get_median_frequency(b)
        
        if f_a is None or f_b is None or f_a <= 0 or f_b <= 0:
            return False
        
        # Check integer ratios
        for n in range(1, 8):
            for m in range(1, 8):
                expected_ratio = n / m
                actual_ratio = f_b / f_a
                
                if abs(actual_ratio - expected_ratio) / expected_ratio < tolerance:
                    return True
        
        return False
    
    def _get_median_frequency(self, ann: Dict) -> Optional[float]:
        """Get median frequency from annotation."""
        # Try points first
        points = ann.get('points', [])
        if points and len(points) > 0:
            freqs = [p[1] for p in points if len(p) >= 2]
            if freqs:
                return float(np.median(freqs))
        
        # Fall back to bounds
        f_min = ann.get('f_min', 0)
        f_max = ann.get('f_max', 0)
        if f_min > 0 and f_max > 0:
            return (f_min + f_max) / 2.0
        
        return None
    
    def _create_event(self, annotations: List[Dict], event_id: int, file_name: str) -> Event:
        """Create Event from list of annotations."""
        if not annotations:
            raise ValueError("Cannot create event from empty annotations")
        
        # Calculate bounds
        t_starts = []
        t_ends = []
        f_mins = []
        f_maxs = []
        annotation_ids = []
        snrs = []
        slopes = []
        harmonic_orders = []
        
        for ann in annotations:
            t_starts.append(min(ann.get('t_start', 0), ann.get('t_end', 0)))
            t_ends.append(max(ann.get('t_start', 0), ann.get('t_end', 0)))
            f_mins.append(min(ann.get('f_min', 0), ann.get('f_max', 0)))
            f_maxs.append(max(ann.get('f_min', 0), ann.get('f_max', 0)))
            annotation_ids.append(ann.get('id'))
            
            if ann.get('snr_db') is not None:
                snrs.append(ann.get('snr_db'))
            if ann.get('slope_hz_per_sec') is not None:
                slopes.append(ann.get('slope_hz_per_sec'))
            if ann.get('harmonic_order') is not None:
                harmonic_orders.append(ann.get('harmonic_order'))
        
        t_start = min(t_starts)
        t_end = max(t_ends)
        f_min = min(f_mins)
        f_max = max(f_maxs)
        
        # Estimate f0 from harmonic relationships
        estimated_f0 = self._estimate_f0(annotations)
        
        # Calculate averages
        avg_snr = np.mean(snrs) if snrs else None
        avg_slope = np.mean(slopes) if slopes else None
        
        # Get label (use most common)
        labels = [ann.get('track_label', '') for ann in annotations if ann.get('track_label')]
        label = max(set(labels), key=labels.count) if labels else ""
        
        return Event(
            event_id=f"EVT_{event_id:03d}",
            file_name=file_name,
            t_start=t_start,
            t_end=t_end,
            duration=t_end - t_start,
            f_min=f_min,
            f_max=f_max,
            annotation_ids=annotation_ids,
            annotation_count=len(annotations),
            estimated_f0=estimated_f0,
            harmonics_detected=sorted(set(harmonic_orders)) if harmonic_orders else [],
            average_snr=float(avg_snr) if avg_snr is not None else None,
            average_slope=float(avg_slope) if avg_slope is not None else None,
            label=label
        )
    
    def _estimate_f0(self, annotations: List[Dict]) -> Optional[float]:
        """Estimate fundamental frequency from harmonic relationships."""
        if not annotations:
            return None
        
        # Get frequencies and harmonic orders
        freq_harmonic_pairs = []
        for ann in annotations:
            f = self._get_median_frequency(ann)
            harmonic_order = ann.get('harmonic_order')
            if f is not None and harmonic_order is not None and harmonic_order > 0:
                freq_harmonic_pairs.append((f, harmonic_order))
        
        if not freq_harmonic_pairs:
            # Try to estimate from frequency ratios
            freqs = [self._get_median_frequency(ann) for ann in annotations]
            freqs = [f for f in freqs if f is not None and f > 0]
            if len(freqs) >= 2:
                # Assume lowest frequency is fundamental
                return min(freqs)
            return None
        
        # Calculate f0 from harmonic relationships: f = n * f0
        f0_candidates = []
        for f, n in freq_harmonic_pairs:
            f0 = f / n
            if f0 > 20:  # Minimum audible frequency
                f0_candidates.append(f0)
        
        if f0_candidates:
            return float(np.median(f0_candidates))
        
        return None

