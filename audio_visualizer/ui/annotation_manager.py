"""
Annotation manager for tracking and managing annotation rectangles.
"""
import logging
from typing import List, Optional, Dict
from pathlib import Path
import json
import csv
from datetime import datetime

from .annotation_data import Annotation

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
        logger.info(f"Cleared {count} annotations")
    
    def set_file_path(self, file_path: str):
        """Update the file path (called when loading a new audio file)."""
        self.file_path = file_path
        self.clear()  # Clear annotations when switching files
    
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
    
    def save_to_json(self) -> bool:
        """Save annotations to JSON file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        json_path = self.get_json_path()
        if not json_path:
            logger.warning("Cannot save annotations: no file path set")
            return False
        
        try:
            # Convert annotations to dictionaries (excluding graphics_handle)
            data = {
                'version': '1.1', # Bump version for Doppler support
                'file_name': Path(self.file_path).name if self.file_path else '',
                'last_modified': datetime.utcnow().isoformat(),
                'annotations': [ann.to_dict() for ann in self.annotations]
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.annotations)} annotations to {json_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving annotations to {json_path}: {e}")
            return False
    
    def load_from_file(self, path: str) -> bool:
        """Load annotations from a JSON or JSONL file.
        
        Args:
            path: Path to JSON or JSONL file
            
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
                    label=record.get('label', ''),
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

    def load_from_json(self) -> bool:
        """Load annotations from JSON file (for current file).
        
        Returns:
            True if loaded successfully, False otherwise
        """
        json_path = self.get_json_path()
        if not json_path:
            return False
        return self.load_from_file(str(json_path))
    
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

    def __len__(self) -> int:
        """Return the number of annotations."""
        return len(self.annotations)
    
    def __iter__(self):
        """Iterate over annotations."""
        return iter(self.annotations)
