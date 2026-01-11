"""
Project Manager
Manages project-based data organization for multi-file analysis.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages project-level data and file organization."""
    
    def __init__(self, project_path: Optional[Path] = None):
        """Initialize project manager.
        
        Args:
            project_path: Path to project directory (None if no project loaded)
        """
        self.project_path = project_path
        self.project_data: Dict = {}
        self.file_managers: Dict[str, any] = {}  # filename -> AnnotationManager reference
        
        # Project structure
        self.files_dir = None
        self.events_dir = None
        self.exports_dir = None
        
        if project_path:
            self._initialize_directories()
    
    def _initialize_directories(self):
        """Initialize project directory structure."""
        if not self.project_path:
            return
        
        self.files_dir = self.project_path / "files"
        self.events_dir = self.project_path / "events"
        self.exports_dir = self.project_path / "exports"
        
        # Create directories if they don't exist
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, path: Path, name: str, description: str = "") -> bool:
        """Create new project structure.
        
        Args:
            path: Directory path for project
            name: Project name
            description: Project description
        
        Returns:
            True if created successfully
        """
        try:
            self.project_path = Path(path)
            self.project_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize directory structure
            self._initialize_directories()
            
            # Create project.json
            self.project_data = {
                "version": "2.0",
                "created": datetime.utcnow().isoformat(),
                "modified": datetime.utcnow().isoformat(),
                "name": name,
                "description": description,
                "files": [],
                "global_settings": {
                    "default_fft_size": 4096,
                    "default_hop_length": 164,
                    "default_window": "hamming",
                    "default_overlap_percent": 96.0
                },
                "sensor_info": {
                    "sensor_id": "",
                    "gps_file": None,
                    "location_description": ""
                }
            }
            
            self._save_project_metadata()
            logger.info(f"Created project: {name} at {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False
    
    def load_project(self, path: Path) -> bool:
        """Load existing project.

        Args:
            path: Path to project directory

        Returns:
            True if loaded successfully
        """
        try:
            self.project_path = Path(path)
            project_json = self.project_path / "project.json"

            if not project_json.exists():
                logger.error(f"Project file not found: {project_json}")
                return False

            with open(project_json, 'r') as f:
                self.project_data = json.load(f)

            # Initialize directories
            self._initialize_directories()

            # Recalculate annotation counts from actual files
            self._recalculate_annotation_counts()

            logger.info(f"Loaded project: {self.project_data.get('name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return False

    def _recalculate_annotation_counts(self):
        """Recalculate annotation counts from actual annotation files."""
        # First, update counts for registered files
        for file_entry in self.project_data.get("files", []):
            filename = file_entry.get("filename")
            annotations_path = self.get_file_annotations_path(filename)

            if annotations_path and annotations_path.exists():
                try:
                    with open(annotations_path, 'r') as f:
                        data = json.load(f)
                    annotation_count = len(data.get("annotations", []))
                    file_entry["annotation_count"] = annotation_count
                    file_entry["analyzed"] = annotation_count > 0
                    logger.debug(f"File {filename}: {annotation_count} annotations")
                except Exception as e:
                    logger.warning(f"Failed to count annotations in {filename}: {e}")
            else:
                file_entry["annotation_count"] = 0
                file_entry["analyzed"] = False

        # Also scan for unregistered annotation files in the files/ directory
        if self.files_dir and self.files_dir.exists():
            registered_files = {f.get("filename") for f in self.project_data.get("files", [])}

            for file_dir in self.files_dir.iterdir():
                if file_dir.is_dir():
                    annotations_json = file_dir / "annotations.json"
                    if annotations_json.exists():
                        # Try to determine the original filename
                        try:
                            with open(annotations_json, 'r') as f:
                                data = json.load(f)
                            original_filename = data.get("file_name", f"{file_dir.name}.wav")

                            # If not registered, add it
                            if original_filename not in registered_files:
                                annotation_count = len(data.get("annotations", []))
                                new_entry = {
                                    "filename": original_filename,
                                    "path": str(file_dir.relative_to(self.project_path)),
                                    "added": datetime.utcnow().isoformat(),
                                    "modified": datetime.utcnow().isoformat(),
                                    "analyzed": annotation_count > 0,
                                    "annotation_count": annotation_count,
                                    "event_count": 0
                                }
                                if "files" not in self.project_data:
                                    self.project_data["files"] = []
                                self.project_data["files"].append(new_entry)
                                logger.info(f"Discovered unregistered file: {original_filename} with {annotation_count} annotations")
                        except Exception as e:
                            logger.warning(f"Failed to process {annotations_json}: {e}")
    
    def save_project(self, auto_export: bool = True) -> bool:
        """Save all project data and optionally export.
        
        Args:
            auto_export: If True, automatically export CSV to exports/ directory
        
        Returns:
            True if saved successfully
        """
        if not self.project_path:
            logger.warning("No project loaded - cannot save")
            return False
        
        try:
            # Update modification time
            self.project_data["modified"] = datetime.utcnow().isoformat()
            
            # Save project metadata
            self._save_project_metadata()
            
            # Auto-export if enabled
            if auto_export:
                self._auto_export_project()
            
            logger.info("Project saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            return False
    
    def _auto_export_project(self):
        """Automatically export project data to exports/ directory."""
        try:
            # Export unified CSV
            csv_path = self.exports_dir / "all_annotations.csv"
            self._export_unified_csv_internal(csv_path)
            logger.info(f"Auto-exported annotations CSV to {csv_path}")
            
            # Export events CSV
            events_csv_path = self.exports_dir / "all_events.csv"
            self._export_events_csv_internal(events_csv_path)
            logger.info(f"Auto-exported events CSV to {events_csv_path}")
            
            # Generate HTML report
            html_path = self.exports_dir / "analysis_report.html"
            self._generate_html_report_internal(html_path)
            logger.info(f"Generated HTML report to {html_path}")
            
        except Exception as e:
            logger.warning(f"Auto-export failed (non-critical): {e}")
    
    def _export_unified_csv_internal(self, output_path: Path):
        """Internal method to export unified CSV.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        all_annotations = []

        # Collect annotations from all files in project
        for file_entry in self.project_data.get("files", []):
            filename = file_entry.get("filename")
            annotations_path = self.get_file_annotations_path(filename)

            if annotations_path and annotations_path.exists():
                try:
                    with open(annotations_path, 'r') as f:
                        data = json.load(f)

                    for ann_dict in data.get("annotations", []):
                        # Add file name to each annotation
                        ann_dict["file_name"] = filename
                        all_annotations.append(ann_dict)

                except Exception as e:
                    logger.warning(f"Failed to load annotations from {filename}: {e}")

        if all_annotations:
            # Get all unique keys across all annotations
            all_keys = set()
            for ann in all_annotations:
                all_keys.update(ann.keys())
            fieldnames = sorted(all_keys)

            # Write CSV using standard library
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_annotations)

            logger.info(f"Exported {len(all_annotations)} annotations to {output_path}")
    
    def _export_events_csv_internal(self, output_path: Path):
        """Internal method to export events CSV.

        Args:
            output_path: Path to output CSV file
        """
        import csv
        from .event_grouper import EventGrouper

        all_annotations = []

        # Collect annotations from all files in project
        for file_entry in self.project_data.get("files", []):
            filename = file_entry.get("filename")
            annotations_path = self.get_file_annotations_path(filename)

            if annotations_path and annotations_path.exists():
                try:
                    with open(annotations_path, 'r') as f:
                        data = json.load(f)

                    for ann_dict in data.get("annotations", []):
                        # Add file name to each annotation
                        ann_dict["file_name"] = filename
                        all_annotations.append(ann_dict)

                except Exception as e:
                    logger.warning(f"Failed to load annotations from {filename}: {e}")

        if not all_annotations:
            logger.warning("No annotations found for events export")
            return

        # Group annotations into events
        grouper = EventGrouper()
        events = grouper.group_annotations(all_annotations)

        if not events:
            logger.warning("No events found to export")
            return

        # Convert events to list of dicts
        events_data = []
        for event in events:
            events_data.append({
                'event_id': event.event_id,
                'file_name': event.file_name,
                't_start': event.t_start,
                't_end': event.t_end,
                'duration': event.duration,
                'f_min': event.f_min,
                'f_max': event.f_max,
                'annotation_count': event.annotation_count,
                'harmonics': ','.join(map(str, event.harmonics_detected)) if event.harmonics_detected else '',
                'estimated_f0': event.estimated_f0,
                'avg_snr': event.average_snr,
                'avg_slope': event.average_slope,
                'label': event.label
            })

        # Write CSV using standard library
        fieldnames = ['event_id', 'file_name', 't_start', 't_end', 'duration',
                      'f_min', 'f_max', 'annotation_count', 'harmonics',
                      'estimated_f0', 'avg_snr', 'avg_slope', 'label']

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events_data)

        logger.info(f"Exported {len(events)} events to {output_path}")
    
    def _generate_html_report_internal(self, output_path: Path):
        """Internal method to generate HTML report.
        
        Args:
            output_path: Path to output HTML file
        """
        from .export_manager import generate_html_report
        from .event_grouper import EventGrouper
        
        # Collect annotations
        all_annotations = []
        for file_entry in self.project_data.get("files", []):
            filename = file_entry.get("filename")
            annotations_path = self.get_file_annotations_path(filename)
            
            if annotations_path and annotations_path.exists():
                try:
                    with open(annotations_path, 'r') as f:
                        data = json.load(f)
                    
                    for ann_dict in data.get("annotations", []):
                        ann_dict["file_name"] = filename
                        all_annotations.append(ann_dict)
                        
                except Exception as e:
                    logger.warning(f"Failed to load annotations from {filename}: {e}")
        
        # Group into events
        grouper = EventGrouper()
        events = grouper.group_annotations(all_annotations)
        
        # Convert events to dicts
        events_data = []
        for event in events:
            events_data.append({
                'event_id': event.event_id,
                'file_name': event.file_name,
                't_start': event.t_start,
                't_end': event.t_end,
                'duration': event.duration,
                'f_min': event.f_min,
                'f_max': event.f_max,
                'annotation_count': event.annotation_count,
                'harmonics': event.harmonics_detected,
                'estimated_f0': event.estimated_f0,
                'avg_snr': event.average_snr,
                'avg_slope': event.average_slope,
                'label': event.label
            })
        
        # Generate report
        project_info = self.get_project_info()
        generate_html_report(project_info, events_data, output_path, len(all_annotations))
    
    def _save_project_metadata(self):
        """Save project.json file."""
        project_json = self.project_path / "project.json"
        with open(project_json, 'w') as f:
            json.dump(self.project_data, f, indent=2)
    
    def add_file(self, audio_path: Path, file_hash: Optional[str] = None,
                  analysis_params: Optional[Dict] = None,
                  audio_properties: Optional[Dict] = None) -> str:
        """Add audio file to project.
        
        Args:
            audio_path: Path to audio file
            file_hash: Optional SHA256 hash of file
            analysis_params: Optional analysis parameters (FFT size, etc.)
            audio_properties: Optional audio file properties (sample_rate, duration, etc.)
        
        Returns:
            File identifier (filename)
        """
        if not self.is_project_loaded():
            logger.warning("Cannot add file: no project loaded")
            return audio_path.name
        
        filename = audio_path.name
        file_stem = audio_path.stem
        
        # Create file directory
        file_dir = self.files_dir / file_stem
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists in project
        existing_file = None
        for f in self.project_data.get("files", []):
            if f.get("filename") == filename:
                existing_file = f
                break
        
        if existing_file:
            # Update existing entry
            existing_file["path"] = str(file_dir.relative_to(self.project_path))
            existing_file["modified"] = datetime.utcnow().isoformat()
            if file_hash:
                existing_file["file_hash"] = file_hash
        else:
            # Add new file entry
            file_entry = {
                "filename": filename,
                "original_path": str(audio_path),
                "path": str(file_dir.relative_to(self.project_path)),
                "added": datetime.utcnow().isoformat(),
                "modified": datetime.utcnow().isoformat(),
                "analyzed": False,
                "annotation_count": 0,
                "event_count": 0
            }
            
            if file_hash:
                file_entry["file_hash"] = file_hash
            
            if "files" not in self.project_data:
                self.project_data["files"] = []
            self.project_data["files"].append(file_entry)
        
        # Save file metadata
        self._save_file_metadata(file_stem, audio_path, analysis_params, 
                                audio_properties=audio_properties,
                                file_hash=file_hash)
        
        return filename
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file.
        
        Args:
            file_path: Path to file
        
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _save_file_metadata(self, file_stem: str, audio_path: Path, 
                           analysis_params: Optional[Dict] = None,
                           audio_properties: Optional[Dict] = None,
                           file_hash: Optional[str] = None):
        """Save metadata for a specific file.
        
        Args:
            file_stem: File stem (without extension)
            audio_path: Path to audio file
            analysis_params: Analysis parameters
            audio_properties: Audio file properties (sample_rate, duration, etc.)
            file_hash: SHA256 hash of file
        """
        if not self.is_project_loaded():
            return
        
        file_dir = self.files_dir / file_stem
        metadata_path = file_dir / "file_metadata.json"
        
        # Try to load existing metadata
        existing_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            except:
                pass
        
        # Update metadata
        metadata = {
            "filename": audio_path.name,
            "original_path": str(audio_path),
            "file_hash": file_hash or existing_metadata.get("file_hash"),
            "audio_properties": audio_properties or existing_metadata.get("audio_properties", {}),
            "analysis_parameters": analysis_params or existing_metadata.get("analysis_parameters", {}),
            "analysis_history": existing_metadata.get("analysis_history", [])
        }
        
        # Add history entry if parameters changed
        if analysis_params and analysis_params != existing_metadata.get("analysis_parameters"):
            history_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "parameters_updated",
                "parameters": analysis_params
            }
            metadata["analysis_history"].append(history_entry)
        
        # Ensure directory exists
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def update_file_stats(self, filename: str, annotation_count: int, event_count: int = 0):
        """Update statistics for a file.
        
        Args:
            filename: File name
            annotation_count: Number of annotations
            event_count: Number of events
        """
        for file_entry in self.project_data.get("files", []):
            if file_entry.get("filename") == filename:
                file_entry["annotation_count"] = annotation_count
                file_entry["event_count"] = event_count
                file_entry["analyzed"] = annotation_count > 0
                file_entry["modified"] = datetime.utcnow().isoformat()
                break
    
    def get_file_annotations_path(self, filename: str, create_if_missing: bool = False) -> Optional[Path]:
        """Get path to annotations.json for a file.

        Args:
            filename: File name
            create_if_missing: If True, return path even if file doesn't exist yet

        Returns:
            Path to annotations.json or None if files_dir is not set
        """
        if not self.files_dir:
            return None

        file_stem = Path(filename).stem
        file_dir = self.files_dir / file_stem
        annotations_path = file_dir / "annotations.json"

        # Return path if exists OR if we want the path for creating new file
        if annotations_path.exists() or create_if_missing:
            return annotations_path

        return None
    
    def get_file_metadata_path(self, filename: str) -> Optional[Path]:
        """Get path to file_metadata.json for a file.
        
        Args:
            filename: File name
        
        Returns:
            Path to file_metadata.json or None
        """
        if not self.is_project_loaded() or not self.files_dir:
            return None
        
        file_stem = Path(filename).stem
        file_dir = self.files_dir / file_stem
        metadata_path = file_dir / "file_metadata.json"
        return metadata_path if metadata_path.exists() else None
    
    def get_project_info(self) -> Dict:
        """Get project information.
        
        Returns:
            Dictionary with project info
        """
        return {
            "name": self.project_data.get("name", "Unknown"),
            "description": self.project_data.get("description", ""),
            "created": self.project_data.get("created"),
            "modified": self.project_data.get("modified"),
            "file_count": len(self.project_data.get("files", [])),
            "total_annotations": sum(f.get("annotation_count", 0) for f in self.project_data.get("files", [])),
            "path": str(self.project_path) if self.project_path else None
        }
    
    def is_project_loaded(self) -> bool:
        """Check if a project is currently loaded.
        
        Returns:
            True if project is loaded
        """
        return self.project_path is not None and (self.project_path / "project.json").exists()

