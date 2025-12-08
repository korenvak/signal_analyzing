"""
Export Manager
Handles advanced export functionality including images with axes and unified CSV.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - image export with axes will be limited")


def export_annotation_image(annotation: Dict,
                            spectrogram_data: np.ndarray,
                            times: np.ndarray,
                            freqs: np.ndarray,
                            output_path: Path,
                            include_track: bool = True,
                            dpi: int = 150,
                            colormap: str = 'magma') -> bool:
    """
    Export annotation as image with proper axes and labels.
    
    Args:
        annotation: Annotation dictionary
        spectrogram_data: 2D spectrogram array (freq x time) in dB
        times: Time axis array
        freqs: Frequency axis array
        output_path: Path to save image
        include_track: Whether to overlay track points
        dpi: Image resolution
        colormap: Colormap name
    
    Returns:
        True if exported successfully
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available for image export with axes")
        return False
    
    try:
        # Extract annotation bounds
        t_min = min(annotation.get('t_start', 0), annotation.get('t_end', 0))
        t_max = max(annotation.get('t_start', 0), annotation.get('t_end', 0))
        f_min = min(annotation.get('f_min', 0), annotation.get('f_max', 0))
        f_max = max(annotation.get('f_min', 0), annotation.get('f_max', 0))
        
        # Find indices
        t_start_idx = np.argmin(np.abs(times - t_min))
        t_end_idx = np.argmin(np.abs(times - t_max))
        f_min_idx = np.argmin(np.abs(freqs - f_min))
        f_max_idx = np.argmin(np.abs(freqs - f_max))
        
        # Ensure valid indices
        t_start_idx = max(0, min(t_start_idx, len(times) - 1))
        t_end_idx = max(t_start_idx + 1, min(t_end_idx + 1, len(times)))
        f_min_idx = max(0, min(f_min_idx, len(freqs) - 1))
        f_max_idx = max(f_min_idx + 1, min(f_max_idx + 1, len(freqs)))
        
        # Extract cutout
        cutout = spectrogram_data[f_min_idx:f_max_idx, t_start_idx:t_end_idx]
        cutout_times = times[t_start_idx:t_end_idx]
        cutout_freqs = freqs[f_min_idx:f_max_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot spectrogram
        extent = [cutout_times[0], cutout_times[-1], cutout_freqs[0], cutout_freqs[-1]]
        im = ax.imshow(cutout, aspect='auto', origin='lower', extent=extent, cmap=colormap)
        
        # Add track if available
        if include_track and annotation.get('points'):
            track_points = annotation['points']
            if len(track_points) > 0:
                track_t = np.array([p[0] for p in track_points])
                track_f = np.array([p[1] for p in track_points])
                
                # Filter points within bounds
                mask = (track_t >= t_min) & (track_t <= t_max) & (track_f >= f_min) & (track_f <= f_max)
                if np.any(mask):
                    ax.plot(track_t[mask], track_f[mask], 'c-', linewidth=2, label='Track')
                    ax.plot(track_t[mask], track_f[mask], 'co', markersize=4)
        
        # Labels and title
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        # Format frequency axis (kHz if > 1000 Hz)
        if f_max > 1000:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x >= 1000 else f'{x:.0f}'))
            ax.set_ylabel('Frequency (kHz)', fontsize=12)
        
        # Title
        title = f"Annotation #{annotation.get('id', '?')}"
        if annotation.get('track_label'):
            title += f" - {annotation['track_label']}"
        if annotation.get('snr_db') is not None:
            title += f" | SNR: {annotation['snr_db']:.1f} dB"
        if annotation.get('slope_hz_per_sec') is not None:
            title += f" | Slope: {annotation['slope_hz_per_sec']:.1f} Hz/s"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        cbar.ax.tick_params(labelsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Exported annotation image to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export annotation image: {e}")
        if HAS_MATPLOTLIB:
            plt.close('all')
        return False


def export_all_annotations_csv(annotations: List[Dict], output_path: Path) -> bool:
    """
    Export all annotations to unified CSV.

    Args:
        annotations: List of annotation dictionaries
        output_path: Path to output CSV file

    Returns:
        True if exported successfully
    """
    import csv

    try:
        if not annotations:
            logger.warning("No annotations to export")
            return False

        # Flatten annotation data
        rows = []
        for ann in annotations:
            row = {
                'id': ann.get('id'),
                'file_name': ann.get('file_name', ''),
                't_start': ann.get('t_start'),
                't_end': ann.get('t_end'),
                'duration': abs(ann.get('t_end', 0) - ann.get('t_start', 0)),
                'f_min': ann.get('f_min'),
                'f_max': ann.get('f_max'),
                'freq_range': abs(ann.get('f_max', 0) - ann.get('f_min', 0)),
                'snr_db': ann.get('snr_db'),
                'slope_hz_per_sec': ann.get('slope_hz_per_sec'),
                'harmonic_order': ann.get('harmonic_order'),
                'event_id': ann.get('event_id'),
                'track_label': ann.get('track_label', ''),
                'point_count': len(ann.get('points', []))
            }
            rows.append(row)

        # Write CSV using standard library
        fieldnames = ['id', 'file_name', 't_start', 't_end', 'duration',
                      'f_min', 'f_max', 'freq_range', 'snr_db', 'slope_hz_per_sec',
                      'harmonic_order', 'event_id', 'track_label', 'point_count']

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} annotations to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        return False


def generate_html_report(project_info: Dict, events: List[Dict], 
                         output_path: Path, annotations_count: int = 0) -> bool:
    """
    Generate HTML report for project.
    
    Args:
        project_info: Project information dictionary
        events: List of event dictionaries
        output_path: Path to output HTML file
        annotations_count: Total number of annotations
    
    Returns:
        True if generated successfully
    """
    try:
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Acoustic Analysis Report - {project_info.get('name', 'Project')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #2a2a2a;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #4a9eff;
            border-bottom: 2px solid #4a9eff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #6ab7ff;
            margin-top: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .info-card {{
            background: #333;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4a9eff;
        }}
        .info-card strong {{
            color: #6ab7ff;
            display: block;
            margin-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #333;
        }}
        th {{
            background: #4a9eff;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #444;
        }}
        tr:hover {{
            background: #3a3a3a;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{
            background: #4caf50;
            color: white;
        }}
        .badge-info {{
            background: #2196f3;
            color: white;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #444;
            text-align: center;
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Acoustic Analysis Report</h1>
        
        <div class="info-grid">
            <div class="info-card">
                <strong>Project Name</strong>
                {project_info.get('name', 'Unknown')}
            </div>
            <div class="info-card">
                <strong>Created</strong>
                {project_info.get('created', 'Unknown')}
            </div>
            <div class="info-card">
                <strong>Last Modified</strong>
                {project_info.get('modified', 'Unknown')}
            </div>
            <div class="info-card">
                <strong>Files Analyzed</strong>
                {project_info.get('file_count', 0)}
            </div>
            <div class="info-card">
                <strong>Total Annotations</strong>
                {annotations_count}
            </div>
            <div class="info-card">
                <strong>Events Detected</strong>
                {len(events)}
            </div>
        </div>
        
        <h2>Events Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Event ID</th>
                    <th>File</th>
                    <th>Time Range</th>
                    <th>Duration (s)</th>
                    <th>Frequency Range (Hz)</th>
                    <th>Annotations</th>
                    <th>Harmonics</th>
                    <th>Est. f0 (Hz)</th>
                    <th>Avg SNR (dB)</th>
                    <th>Label</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for event in events:
            harmonics_str = ', '.join(map(str, event.get('harmonics', []))) if event.get('harmonics') else '-'
            # Format values safely
            est_f0 = event.get('estimated_f0')
            est_f0_str = f"{est_f0:.1f}" if est_f0 is not None else '-'
            avg_snr = event.get('avg_snr')
            avg_snr_str = f"{avg_snr:.1f}" if avg_snr is not None else '-'
            label = event.get('label', '') or '-'

            html_content += f"""
                <tr>
                    <td><span class="badge badge-info">{event.get('event_id', 'N/A')}</span></td>
                    <td>{event.get('file_name', 'N/A')}</td>
                    <td>{event.get('t_start', 0):.2f} - {event.get('t_end', 0):.2f}</td>
                    <td>{event.get('duration', 0):.2f}</td>
                    <td>{event.get('f_min', 0):.0f} - {event.get('f_max', 0):.0f}</td>
                    <td>{event.get('annotation_count', 0)}</td>
                    <td>{harmonics_str}</td>
                    <td>{est_f0_str}</td>
                    <td>{avg_snr_str}</td>
                    <td>{label}</td>
                </tr>
"""
        
        html_content += f"""
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Audio Visualizer - Acoustic Analysis Tool</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return False

