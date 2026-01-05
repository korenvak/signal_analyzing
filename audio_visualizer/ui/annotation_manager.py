"""
Annotation manager for tracking and managing annotation rectangles.
"""
import logging
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json
import csv
from datetime import datetime
import numpy as np

from .annotation_data import Annotation
from ..core.auto_detector import AutomaticTrackDetector, DetectedTrack

logger = logging.getLogger(__name__)


class AnnotationManager:
    """Manages a collection of annotations for a single audio file."""

    def __init__(self, file_path: Optional[str] = None):
        """Initialize annotation manager.

        Args:
            file_path: Path to the audio file (used for loading/saving annotations)
        """
        self.file_path = file_path
        self.annotations: List[Annotation] = []
        self.next_id = 1

        # Undo/Redo stacks: each entry is ('action', data)
        # Actions: 'add' (data=annotation), 'remove' (data=annotation), 'modify' (data=(old, new))
        self._undo_stack: List[Tuple[str, any]] = []
        self._redo_stack: List[Tuple[str, any]] = []
        self._max_undo = 50
    
    def add_annotation(self, annotation: Annotation) -> Annotation:
        """Add a new annotation.
        
        Args:
            annotation: The annotation to add (ID will be auto-assigned if needed)
        
        Returns:
            The annotation with its ID set
        """
        if annotation.id is None or annotation.id == 0:
            annotation.id = self.next_id
            self.next_id += 1
        
        # Ensure ID is unique
        existing_ids = {ann.id for ann in self.annotations}
        while annotation.id in existing_ids:
            annotation.id = self.next_id
            self.next_id += 1
        
        self.annotations.append(annotation)
        logger.info(f"Added annotation {annotation.id}: "
                   f"time=[{annotation.t_start:.3f}, {annotation.t_end:.3f}]s, "
                   f"freq=[{annotation.f_min:.1f}, {annotation.f_max:.1f}]Hz")

        # Push to undo stack (undo of 'add' is remove)
        self._push_undo('add', annotation)

        return annotation
    
    def remove_annotation(self, annotation_id: int) -> bool:
        """Remove an annotation by ID.
        
        Args:
            annotation_id: ID of the annotation to remove
        
        Returns:
            True if annotation was found and removed, False otherwise
        """
        for i, ann in enumerate(self.annotations):
            if ann.id == annotation_id:
                removed = self.annotations.pop(i)
                logger.info(f"Removed annotation {annotation_id}")
                # Push to undo stack (undo of 'remove' is add back)
                self._push_undo('remove', removed)
                return True
        return False
    
    def get_annotation(self, annotation_id: int) -> Optional[Annotation]:
        """Get an annotation by ID.
        
        Args:
            annotation_id: ID of the annotation to retrieve
        
        Returns:
            The annotation if found, None otherwise
        """
        for ann in self.annotations:
            if ann.id == annotation_id:
                return ann
        return None
    
    def get_annotation_at_point(self, time: float, freq: float) -> Optional[Annotation]:
        """Get the annotation containing a specific point.
        
        Args:
            time: Time coordinate in seconds
            freq: Frequency coordinate in Hz
        
        Returns:
            The first annotation containing the point, or None
        """
        # Return the last annotation (topmost) that contains the point
        for ann in reversed(self.annotations):
            if ann.contains_point(time, freq):
                return ann
        return None
    
    def clear(self):
        """Clear all annotations."""
        count = len(self.annotations)
        self.annotations.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.info(f"Cleared {count} annotations")

    # ==================== Undo/Redo ====================

    def _push_undo(self, action: str, data):
        """Push an action to the undo stack."""
        self._undo_stack.append((action, data))
        # Limit stack size
        while len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        # Clear redo stack on new action
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[Tuple[str, any]]:
        """Undo the last annotation action.

        Returns:
            Tuple of (action_type, annotation) that was undone, or None if nothing to undo.
            action_type is 'add' or 'remove' indicating what was UNDONE (reverse of original).
        """
        if not self._undo_stack:
            logger.info("Nothing to undo")
            return None

        action, data = self._undo_stack.pop()

        if action == 'add':
            # Undo an add = remove the annotation
            for i, ann in enumerate(self.annotations):
                if ann.id == data.id:
                    self.annotations.pop(i)
                    self._redo_stack.append(('add', data))
                    logger.info(f"Undid add: removed annotation {data.id}")
                    return ('remove', data)
        elif action == 'remove':
            # Undo a remove = add the annotation back
            self.annotations.append(data)
            self._redo_stack.append(('remove', data))
            logger.info(f"Undid remove: restored annotation {data.id}")
            return ('add', data)

        return None

    def redo(self) -> Optional[Tuple[str, any]]:
        """Redo the last undone annotation action.

        Returns:
            Tuple of (action_type, annotation) that was redone, or None if nothing to redo.
        """
        if not self._redo_stack:
            logger.info("Nothing to redo")
            return None

        action, data = self._redo_stack.pop()

        if action == 'add':
            # Redo an add = add the annotation again
            self.annotations.append(data)
            self._undo_stack.append(('add', data))
            logger.info(f"Redid add: restored annotation {data.id}")
            return ('add', data)
        elif action == 'remove':
            # Redo a remove = remove the annotation again
            for i, ann in enumerate(self.annotations):
                if ann.id == data.id:
                    self.annotations.pop(i)
                    self._undo_stack.append(('remove', data))
                    logger.info(f"Redid remove: removed annotation {data.id}")
                    return ('remove', data)

        return None

    def set_file_path(self, file_path: str, clear_annotations: bool = False):
        """Update the file path (called when loading a new audio file).

        Args:
            file_path: Path to the new audio file
            clear_annotations: If True, clear existing annotations (default: False to persist across files)
        """
        self.file_path = file_path
        if clear_annotations:
            self.clear()  # Only clear if explicitly requested
    
    def get_json_path(self) -> Optional[Path]:
        """Get the path to the JSON file for saving/loading annotations.
        
        Returns:
            Path to JSON file, or None if file_path is not set
        """
        if not self.file_path:
            return None
        
        audio_path = Path(self.file_path)
        json_path = audio_path.parent / f"{audio_path.stem}_annotations.json"
        return json_path
    
    def save_to_json(self, spectrogram_params: Optional[Dict] = None, 
                     project_manager=None) -> bool:
        """Save annotations to JSON file with metadata.
        
        Args:
            spectrogram_params: Optional dict with FFT parameters (fft_size, hop_length, window, etc.)
            project_manager: Optional ProjectManager instance for project-based saving
        
        Returns:
            True if saved successfully, False otherwise
        """
        # Determine save path
        if project_manager and project_manager.is_project_loaded():
            # Use project structure - create_if_missing=True to get path for new files
            filename = Path(self.file_path).name if self.file_path else 'unknown'
            json_path = project_manager.get_file_annotations_path(filename, create_if_missing=True)
        else:
            # Use legacy path (next to audio file)
            json_path = self.get_json_path()
        
        if not json_path:
            logger.warning("Cannot save annotations: no file path set")
            return False
        
        try:
            # Convert annotations to dictionaries (excluding graphics_handle)
            data = {
                'version': '2.0',  # Bump version for project support
                'file_name': Path(self.file_path).name if self.file_path else '',
                'last_modified': datetime.utcnow().isoformat(),
                'annotations': [ann.to_dict() for ann in self.annotations]
            }
            
            # Add spectrogram parameters if provided
            if spectrogram_params:
                data['spectrogram_params'] = {
                    'fft_size': spectrogram_params.get('fft_size'),
                    'hop_length': spectrogram_params.get('hop_length'),
                    'window': spectrogram_params.get('window_type') or spectrogram_params.get('window'),
                    'overlap_percent': spectrogram_params.get('overlap_percent'),
                    'sample_rate': spectrogram_params.get('sample_rate')
                }
            
            # Ensure parent directory exists
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.annotations)} annotations to {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotations to {json_path}: {e}")
            return False
    
    def load_from_file(self, path: str, project_manager=None) -> bool:
        """Load annotations from a JSON or JSONL file.
        
        Args:
            path: Path to JSON or JSONL file
            project_manager: Optional ProjectManager instance (for compatibility)
            
        Returns:
            True if successful
        """
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return False
                
            # Clear existing
            self.annotations.clear()
            
            # Detect format by extension or content
            if path_obj.suffix.lower() == '.jsonl':
                # JSONL format: one JSON object per line
                with open(path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            # Try to convert cutout manifest format to annotation
                            annotation = self._record_to_annotation(record)
                            if annotation:
                                self.annotations.append(annotation)
                                if annotation.id >= self.next_id:
                                    self.next_id = annotation.id + 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
            else:
                # Standard JSON format
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load from annotations list
                for ann_dict in data.get('annotations', []):
                    annotation = Annotation.from_dict(ann_dict)
                    self.annotations.append(annotation)
                    if annotation.id >= self.next_id:
                        self.next_id = annotation.id + 1
            
            logger.info(f"Loaded {len(self.annotations)} annotations from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading annotations from {path}: {e}")
            return False
    
    def _record_to_annotation(self, record: dict) -> Optional[Annotation]:
        """Convert a cutout manifest record to an Annotation.
        
        Args:
            record: Dictionary from JSONL line
            
        Returns:
            Annotation object or None if conversion fails
        """
        try:
            # Handle cutout manifest format
            if 'time_start' in record and 'freq_low' in record:
                ann_id = record.get('annotation_id', record.get('rect_id', self.next_id))
                return Annotation(
                    id=ann_id,
                    t_start=record['time_start'],
                    t_end=record['time_end'],
                    f_min=record['freq_low'],
                    f_max=record['freq_high'],
                    track_label=record.get('label', ''),
                    file_name=record.get('file_name', '')
                )
            # Handle standard annotation format
            elif 't_start' in record:
                return Annotation.from_dict(record)
            else:
                return None
        except (KeyError, TypeError) as e:
            logger.debug(f"Could not convert record to annotation: {e}")
            return None

    def load_from_json(self, project_manager=None) -> bool:
        """Load annotations from JSON file (for current file).
        
        Args:
            project_manager: Optional ProjectManager instance
        
        Returns:
            True if loaded successfully, False otherwise
        """
        # Try project path first if project is loaded
        if project_manager and project_manager.is_project_loaded() and self.file_path:
            filename = Path(self.file_path).name
            project_path = project_manager.get_file_annotations_path(filename)
            if project_path and project_path.exists():
                return self.load_from_file(str(project_path), project_manager=project_manager)
        
        # Fall back to legacy path (next to audio file)
        json_path = self.get_json_path()
        if not json_path:
            return False
        return self.load_from_file(str(json_path), project_manager=project_manager)
    
    def export_project_csv(self, root_dir: str, output_path: str) -> bool:
        """
        Scan directory for all annotation JSON files and export a master CSV.
        
        Args:
            root_dir: Directory to scan recursively
            output_path: Path to save the CSV
            
        Returns:
            True if successful
        """
        try:
            root = Path(root_dir)
            all_annotations = []
            
            # Find all _annotations.json files
            json_files = list(root.rglob("*_annotations.json"))
            logger.info(f"Found {len(json_files)} annotation files in {root_dir}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    file_name = data.get('file_name', json_file.stem.replace('_annotations', ''))
                    
                    for ann_dict in data.get('annotations', []):
                        # Flatten for CSV
                        flat = {
                            'File': file_name,
                            'ID': ann_dict.get('id'),
                            'Time Start (s)': f"{ann_dict.get('t_start', 0):.3f}",
                            'Time End (s)': f"{ann_dict.get('t_end', 0):.3f}",
                            'Freq Min (Hz)': f"{ann_dict.get('f_min', 0):.1f}",
                            'Freq Max (Hz)': f"{ann_dict.get('f_max', 0):.1f}",
                            'Label': ann_dict.get('track_label', ''),
                            'Approved': ann_dict.get('is_approved', False)
                        }
                        
                        # Add Doppler results if available
                        doppler = ann_dict.get('doppler_result')
                        if doppler:
                            flat.update({
                                'Velocity (m/s)': f"{doppler.get('velocity', 0):.2f}",
                                'Velocity (km/h)': f"{doppler.get('velocity', 0) * 3.6:.2f}",
                                'CPA Dist (m)': f"{doppler.get('cpa_distance', 0):.2f}",
                                'Time CPA (s)': f"{doppler.get('t_cpa', 0):.3f}",
                                'f0 (Hz)': f"{doppler.get('f0', 0):.1f}",
                                'RMSE': f"{doppler.get('rmse', 0):.2f}"
                            })
                        else:
                            flat.update({
                                'Velocity (m/s)': '',
                                'Velocity (km/h)': '',
                                'CPA Dist (m)': '',
                                'Time CPA (s)': '',
                                'f0 (Hz)': '',
                                'RMSE': ''
                            })
                            
                        all_annotations.append(flat)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {json_file}: {e}")
            
            if not all_annotations:
                logger.warning("No annotations found to export")
                return False
                
            # Write CSV
            fieldnames = [
                'File', 'ID', 'Label', 'Approved',
                'Time Start (s)', 'Time End (s)', 'Freq Min (Hz)', 'Freq Max (Hz)',
                'Velocity (km/h)', 'CPA Dist (m)', 'Time CPA (s)', 'f0 (Hz)', 'RMSE'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_annotations)
                
            logger.info(f"Exported master CSV to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export project CSV: {e}")
            return False

    def auto_detect_track(self, annotation_id: int,
                          spectrogram: np.ndarray,
                          times: np.ndarray,
                          freqs: np.ndarray,
                          sample_rate: float = 44100.0,
                          hop_length: int = 512) -> List[DetectedTrack]:
        """Run automatic track detection within an annotation region.

        Args:
            annotation_id: ID of the annotation to detect tracks in
            spectrogram: Full spectrogram data (freq x time)
            times: Time values for spectrogram columns
            freqs: Frequency values for spectrogram rows
            sample_rate: Audio sample rate
            hop_length: FFT hop length

        Returns:
            List of DetectedTrack objects (sorted by score, best first)
        """
        annotation = self.get_annotation(annotation_id)
        if annotation is None:
            logger.warning(f"Annotation {annotation_id} not found")
            return []

        detector = AutomaticTrackDetector()

        try:
            tracks = detector.detect_in_region(
                spectrogram=spectrogram,
                times=times,
                freqs=freqs,
                t_start=annotation.t_start,
                t_end=annotation.t_end,
                f_min=annotation.f_min,
                f_max=annotation.f_max,
                sample_rate=sample_rate,
                hop_length=hop_length
            )
            logger.info(f"Auto-detected {len(tracks)} tracks in annotation {annotation_id}")
            return tracks
        except Exception as e:
            logger.error(f"Auto-detection failed for annotation {annotation_id}: {e}")
            return []

    def set_annotation_track(self, annotation_id: int,
                             track_points: List[Tuple[float, float]]) -> bool:
        """Set the track points for an annotation (from auto-detect or manual drawing).

        Args:
            annotation_id: ID of the annotation to update
            track_points: List of (time, freq) tuples representing the track

        Returns:
            True if successful, False otherwise
        """
        annotation = self.get_annotation(annotation_id)
        if annotation is None:
            logger.warning(f"Annotation {annotation_id} not found")
            return False

        # Set the points
        annotation.points = list(track_points)
        logger.info(f"Set {len(track_points)} track points for annotation {annotation_id}")
        return True

    def get_annotation_track(self, annotation_id: int) -> Optional[List[Tuple[float, float]]]:
        """Get the track points for an annotation.

        Args:
            annotation_id: ID of the annotation

        Returns:
            List of (time, freq) tuples, or None if not found/no track
        """
        annotation = self.get_annotation(annotation_id)
        if annotation is None:
            return None
        return annotation.points if annotation.points else None

    def __len__(self) -> int:
        """Return the number of annotations."""
        return len(self.annotations)

    def __iter__(self):
        """Iterate over annotations."""
        return iter(self.annotations)
