"""
Automatic track detection using Meijering ridge filter and DBSCAN clustering.

Adapted from doppler_physics_v2.py for integration with the audio visualizer.
Enhanced with gap bridging and track merging for handling discontinuous tracks.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

try:
    from skimage.filters import meijering
    from skimage.morphology import binary_closing, binary_opening, skeletonize, binary_dilation, binary_erosion
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.signal import savgol_filter
    from scipy.ndimage import median_filter
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Rectangle footprint for morphology - compatible with different skimage versions
try:
    from skimage.morphology import footprint_rectangle
    def make_rect(h, w):
        return footprint_rectangle((h, w))
except ImportError:
    try:
        from skimage.morphology import rectangle
        def make_rect(h, w):
            return rectangle(h, w)
    except ImportError:
        def make_rect(h, w):
            return np.ones((h, w), dtype=np.uint8)

logger = logging.getLogger(__name__)


@dataclass
class DetectedTrack:
    """A detected track from automatic detection."""
    points: List[Tuple[float, float]]  # (time, freq) world coordinates
    score: float  # Higher is better
    duration: float  # Duration in seconds
    freq_range: Tuple[float, float]  # (f_min, f_max)
    point_count: int  # Number of points in the track
    merged_from: List[int] = field(default_factory=list)  # Track indices that were merged into this

    def __repr__(self):
        merged_info = f", merged={len(self.merged_from)}" if self.merged_from else ""
        return (f"DetectedTrack(score={self.score:.2f}, duration={self.duration:.2f}s, "
                f"freq_range={self.freq_range[0]:.0f}-{self.freq_range[1]:.0f}Hz, "
                f"points={self.point_count}{merged_info})")


class AutomaticTrackDetector:
    """
    Detects Doppler tracks within annotation regions using Meijering ridge filter
    and DBSCAN clustering.

    Pipeline (adapted from doppler_physics_v2.py):
    1. Extract region from spectrogram
    2. Normalize to [0,1]
    3. Apply high-pass (zero low freq bins)
    4. Median subtraction (horizontal noise)
    5. Vertical impulsive noise removal
    6. Meijering ridge filter
    7. Threshold at percentile
    8. DBSCAN clustering
    9. Geometric filtering (duration, point count)
    10. Skeletonize + extract centerline
    11. Savgol smoothing
    12. Return tracks sorted by score
    """

    def __init__(self):
        # Parameters - tuned for Doppler track detection
        # Using wider sigma range to capture both steep and shallow ridges
        self.meijering_sigmas = range(1, 6)  # Extended range for steep ridges
        self.dbscan_eps = 0.25  # Increased for better connectivity on steep slopes
        self.dbscan_min_samples = 8  # Reduced to capture more ridge points
        self.min_track_duration_sec = 0.10  # Reduced for partial track detection
        self.min_track_points = 15  # Reduced for smaller segments
        self.high_pass_hz = 50.0  # Lowered to catch lower frequency tracks
        self.threshold_percentile = 97.0  # Lowered to capture more ridge pixels (was 99.6)
        self.savgol_window = 11
        self.savgol_poly = 2
        self.min_skeleton_length = 20  # Reduced for smaller segments

        # Gap bridging parameters - more aggressive for Doppler tracks
        self.gap_threshold_sec = 1.0  # Increased to bridge larger gaps
        self.gap_freq_tolerance_hz = 300.0  # Increased for steep Doppler slopes
        self.gap_slope_tolerance = 1500.0  # Much higher for steep Doppler curves (can be 50+ Hz/s)
        self.enable_gap_bridging = True

        # Morphological gap bridging - use both horizontal and diagonal
        self.morph_close_width = 9  # Wider for better horizontal bridging
        self.morph_close_height = 3  # Add some vertical extent for diagonal bridging

        # Track merging parameters
        self.merge_similar_tracks = True
        self.merge_freq_tolerance = 150.0  # Increased tolerance
        self.merge_slope_tolerance = 800.0  # Increased for steep tracks

        # Check dependencies
        if not SKIMAGE_AVAILABLE:
            logger.warning("scikit-image not available - auto detection will be limited")
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - DBSCAN clustering disabled")
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available - smoothing disabled")

    def detect_in_region(self,
                         spectrogram: np.ndarray,
                         times: np.ndarray,
                         freqs: np.ndarray,
                         t_start: float,
                         t_end: float,
                         f_min: float,
                         f_max: float,
                         sample_rate: float = 44100.0,
                         hop_length: int = 512) -> List[DetectedTrack]:
        """
        Detect tracks within annotation region.

        Args:
            spectrogram: Full spectrogram data (2D array, freq x time)
            times: Time values for spectrogram columns
            freqs: Frequency values for spectrogram rows
            t_start, t_end: Time bounds of annotation
            f_min, f_max: Frequency bounds of annotation
            sample_rate: Audio sample rate
            hop_length: FFT hop length

        Returns:
            List of DetectedTrack objects, sorted by score (best first)
        """
        # 1. Extract region
        region, region_times, region_freqs = self._extract_region(
            spectrogram, times, freqs, t_start, t_end, f_min, f_max
        )

        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            logger.warning("Region too small for detection")
            return []

        logger.info(f"Detecting in region: shape={region.shape}, "
                   f"t=[{t_start:.2f}, {t_end:.2f}], f=[{f_min:.0f}, {f_max:.0f}]")

        # Use Meijering + DBSCAN method (from doppler_physics_v2.py)
        if not SKIMAGE_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Detection unavailable (missing skimage/sklearn)")
            return []

        # 2. Preprocess
        processed = self._preprocess(region, region_freqs)

        # 3. Ridge detection
        ridges = self._detect_ridges(processed)

        # 4. Threshold
        binary_mask = self._threshold(ridges)

        logger.info(f"Ridge detection: {binary_mask.sum()} points above threshold")

        if binary_mask.sum() < self.min_track_points:
            logger.info("Not enough ridge points found")
            return []

        # 5. Cluster
        clusters = self._cluster_points(binary_mask)
        logger.info(f"DBSCAN found {len(clusters)} clusters")

        if not clusters:
            logger.info("No clusters found after DBSCAN")
            return []

        # 6. Geometric filtering + build clean_mask
        clean_mask = np.zeros_like(binary_mask, dtype=bool)
        valid_cluster_count = 0

        # Calculate dt for geometric filtering
        if len(region_times) > 1:
            dt = (region_times[-1] - region_times[0]) / (len(region_times) - 1)
        else:
            dt = hop_length / sample_rate if sample_rate > 0 else 0.01

        # Adaptive thresholds for small annotation regions
        # Scale based on number of time frames (more reliable than duration)
        n_time_frames = region.shape[1]
        if n_time_frames < 50:
            # Very small region
            min_points_threshold = 5
            min_duration_threshold = 0.02
        elif n_time_frames < 100:
            min_points_threshold = 8
            min_duration_threshold = 0.05
        elif n_time_frames < 200:
            min_points_threshold = 12
            min_duration_threshold = 0.08
        elif n_time_frames < 500:
            min_points_threshold = 18
            min_duration_threshold = 0.12
        else:
            min_points_threshold = self.min_track_points  # 25
            min_duration_threshold = self.min_track_duration_sec  # 0.15

        region_duration = region_times[-1] - region_times[0] if len(region_times) > 1 else 0
        logger.info(f"Geometric filter thresholds: min_points={min_points_threshold}, "
                   f"min_duration={min_duration_threshold}s (n_frames={n_time_frames}, duration={region_duration:.2f}s)")

        for i, cluster in enumerate(clusters):
            n_points = len(cluster)

            if n_points < min_points_threshold:
                logger.debug(f"Cluster {i}: rejected ({n_points} points < {min_points_threshold})")
                continue

            # Get bounding box in indices
            min_f_idx, min_t_idx = np.min(cluster, axis=0)
            max_f_idx, max_t_idx = np.max(cluster, axis=0)

            height = max_f_idx - min_f_idx
            width = max_t_idx - min_t_idx + 1

            # Calculate duration in seconds
            duration_sec = width * dt

            # Filter A: long enough in time
            if duration_sec < min_duration_threshold:
                logger.debug(f"Cluster {i}: rejected (duration {duration_sec:.3f}s < {min_duration_threshold}s)")
                continue

            # Filter B: not purely vertical (impulsive noise)
            if width < 3 and height > 20:
                logger.debug(f"Cluster {i}: rejected (vertical noise)")
                continue

            # Add to clean mask
            clean_mask[cluster[:, 0], cluster[:, 1]] = True
            valid_cluster_count += 1
            logger.debug(f"Cluster {i}: ACCEPTED ({n_points} points, duration={duration_sec:.3f}s)")

        logger.info(f"Geometric filtering: kept {valid_cluster_count}/{len(clusters)} clusters")

        if not clean_mask.any():
            logger.info("No valid clusters after geometric filtering")
            return []

        # 7. Morphology to extract tracks (same as doppler_physics_v2.py)
        # Use adaptive min_skeleton_length based on region size
        # doppler_physics_v2.py uses min_length=30 for full spectrograms (thousands of frames)
        # Scale proportionally for smaller regions
        n_time_frames = region.shape[1]
        if n_time_frames < 50:
            min_skel_length = 3  # Very small region
        elif n_time_frames < 100:
            min_skel_length = 5
        elif n_time_frames < 200:
            min_skel_length = 8
        elif n_time_frames < 500:
            min_skel_length = 15
        else:
            min_skel_length = self.min_skeleton_length  # 30

        skeleton_tracks = self._extract_tracks_morphology(clean_mask, min_length=min_skel_length)
        logger.info(f"Morphology: {len(skeleton_tracks)} skeleton tracks extracted (min_length={min_skel_length}, n_frames={n_time_frames})")

        if not skeleton_tracks:
            logger.info("No tracks after morphology")
            return []

        # 8. Smooth tracks and convert to world coordinates
        raw_tracks = []
        for i, coords in enumerate(skeleton_tracks):
            # coords is (freq_idx, time_idx) pairs sorted by time
            # Convert to world coordinates
            centerline = self._coords_to_world(coords, region_times, region_freqs)

            if len(centerline) < 3:
                continue

            # Smooth the track
            smoothed = self._smooth_track(centerline)

            if len(smoothed) < 3:
                continue

            # Calculate score
            score = self._calculate_track_score(smoothed)

            # Get track properties
            t_vals = [p[0] for p in smoothed]
            f_vals = [p[1] for p in smoothed]
            duration = max(t_vals) - min(t_vals)
            freq_range = (min(f_vals), max(f_vals))

            track = DetectedTrack(
                points=smoothed,
                score=score,
                duration=duration,
                freq_range=freq_range,
                point_count=len(smoothed),
                merged_from=[i]  # Track original index
            )
            raw_tracks.append(track)

        logger.info(f"Raw tracks before gap bridging: {len(raw_tracks)}")

        # 9. Bridge gaps between track segments (new step)
        if self.enable_gap_bridging and len(raw_tracks) > 1:
            tracks = self._bridge_track_gaps(raw_tracks)
            logger.info(f"After gap bridging: {len(tracks)} tracks")
        else:
            tracks = raw_tracks

        # 10. Merge similar parallel tracks (optional)
        if self.merge_similar_tracks and len(tracks) > 1:
            tracks = self._merge_similar_tracks(tracks)
            logger.info(f"After merging similar tracks: {len(tracks)} tracks")

        # Sort by score (highest first)
        tracks.sort(key=lambda t: t.score, reverse=True)

        logger.info(f"Detected {len(tracks)} tracks")
        return tracks

    def _extract_region(self,
                        spectrogram: np.ndarray,
                        times: np.ndarray,
                        freqs: np.ndarray,
                        t_start: float,
                        t_end: float,
                        f_min: float,
                        f_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract spectrogram region within annotation bounds."""
        # Find time indices
        t_mask = (times >= t_start) & (times <= t_end)
        t_indices = np.where(t_mask)[0]

        # Find frequency indices
        f_mask = (freqs >= f_min) & (freqs <= f_max)
        f_indices = np.where(f_mask)[0]

        if len(t_indices) == 0 or len(f_indices) == 0:
            return np.array([]), np.array([]), np.array([])

        # Extract region
        t_start_idx = t_indices[0]
        t_end_idx = t_indices[-1] + 1
        f_start_idx = f_indices[0]
        f_end_idx = f_indices[-1] + 1

        region = spectrogram[f_start_idx:f_end_idx, t_start_idx:t_end_idx]
        region_times = times[t_start_idx:t_end_idx]
        region_freqs = freqs[f_start_idx:f_end_idx]

        return region, region_times, region_freqs

    def _preprocess(self, region: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Preprocess spectrogram region for ridge detection."""
        # Make a copy
        processed = region.copy().astype(np.float64)

        logger.info(f"Preprocess input: shape={processed.shape}, "
                   f"value range=[{processed.min():.2f}, {processed.max():.2f}]")

        # Handle dB values - convert to linear if needed (check if values are negative)
        if processed.min() < 0:
            # Likely in dB, normalize to [0, 1]
            processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-10)
        else:
            # Normalize to [0, 1]
            processed = processed / (processed.max() + 1e-10)

        logger.info(f"After normalization: value range=[{processed.min():.4f}, {processed.max():.4f}]")

        # Apply high-pass (zero out low frequency bins) - but only if region includes low freqs
        if len(freqs) > 0 and self.high_pass_hz > 0 and freqs[0] < self.high_pass_hz:
            hp_idx = np.searchsorted(freqs, self.high_pass_hz)
            if hp_idx > 0 and hp_idx < len(freqs):
                processed[:hp_idx, :] = 0.0
                logger.info(f"High-pass: zeroed {hp_idx} low-freq bins (below {self.high_pass_hz}Hz)")

        # Remove horizontal noise (median over time)
        bg_horizontal = np.median(processed, axis=1, keepdims=True)
        processed = np.maximum(processed - bg_horizontal, 0.0)

        logger.info(f"After horizontal median subtraction: value range=[{processed.min():.4f}, {processed.max():.4f}]")

        # Remove vertical impulsive noise (median over frequency)
        bg_vertical = np.median(processed, axis=0, keepdims=True)
        processed = np.maximum(processed - 2.0 * bg_vertical, 0.0)

        logger.info(f"After vertical noise removal: value range=[{processed.min():.4f}, {processed.max():.4f}], "
                   f"non-zero pixels={np.count_nonzero(processed)}")

        return processed

    def _detect_ridges(self, processed: np.ndarray) -> np.ndarray:
        """Apply Meijering ridge filter - simple and robust approach."""
        if not SKIMAGE_AVAILABLE:
            logger.warning("skimage not available, skipping Meijering filter")
            return processed

        try:
            logger.info(f"Running Meijering filter with sigmas={list(self.meijering_sigmas)}")

            # Apply Meijering filter - it detects ridges at all orientations
            ridges = meijering(processed, sigmas=self.meijering_sigmas, black_ridges=False)

            # Normalize
            ridges = (ridges - ridges.min()) / (ridges.max() - ridges.min() + 1e-10)
            logger.info(f"Meijering output: value range=[{ridges.min():.4f}, {ridges.max():.4f}]")
            return ridges
        except Exception as e:
            logger.error(f"Meijering filter failed: {e}")
            return processed

    def _threshold(self, ridges: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding based on region size."""
        n_pixels = ridges.size

        # Adaptive threshold based on region size
        # Larger regions can use higher percentile (more selective)
        if n_pixels < 5000:
            percentile = 95.0  # Lenient for small regions
        elif n_pixels < 20000:
            percentile = 97.0
        elif n_pixels < 50000:
            percentile = 98.0
        else:
            percentile = 99.0  # More selective for large regions

        thresh = np.percentile(ridges, percentile)
        binary_mask = ridges > thresh

        logger.info(f"Threshold: percentile={percentile:.1f}%, thresh={thresh:.4f}, "
                   f"pixels above={binary_mask.sum()} / {n_pixels}")

        return binary_mask

    def _cluster_points(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """Cluster ridge points using DBSCAN with adaptive parameters."""
        points = np.column_stack(np.where(binary_mask))

        if len(points) < 5:
            logger.info(f"Too few points for clustering: {len(points)}")
            return []

        # Try DBSCAN first if available
        if SKLEARN_AVAILABLE:
            # Scale points for DBSCAN - use separate scaling for freq and time
            # to handle steep Doppler tracks better
            scaler = StandardScaler()
            points_scaled = scaler.fit_transform(points)

            # Adaptive eps based on number of points
            if len(points) < 50:
                eps = 0.5  # Lenient for few points
                min_samples = 3
            elif len(points) < 200:
                eps = 0.35
                min_samples = 5
            elif len(points) < 500:
                eps = 0.25
                min_samples = 8
            else:
                eps = 0.18  # Original value
                min_samples = 15

            logger.info(f"DBSCAN params: eps={eps}, min_samples={min_samples}, n_points={len(points)}")

            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_scaled)
            labels = db.labels_

            # Group points by cluster
            clusters = []
            unique_labels = set(labels)
            n_noise = np.sum(labels == -1)
            logger.info(f"DBSCAN found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters, {n_noise} noise points")

            for k in unique_labels:
                if k == -1:  # Noise
                    continue
                cluster_mask = labels == k
                cluster_points = points[cluster_mask]
                clusters.append(cluster_points)
                logger.debug(f"  Cluster {k}: {len(cluster_points)} points")

            if clusters:
                return clusters
            else:
                logger.info("DBSCAN found no clusters, falling back to connected components")

        # Fallback: use connected components (label from skimage)
        if SKIMAGE_AVAILABLE:
            logger.info("Using connected components as fallback clustering")
            labels_img = label(binary_mask, connectivity=2)
            regions = regionprops(labels_img)

            clusters = []
            for r in regions:
                coords = r.coords  # (N, 2) array
                if len(coords) >= 5:  # Minimum cluster size
                    clusters.append(coords)

            logger.info(f"Connected components found {len(clusters)} clusters")
            return clusters

        return []

    def _extract_tracks_morphology(self, clean_mask: np.ndarray, min_length: int = None) -> List[np.ndarray]:
        """
        Extract tracks from clean binary mask using morphology.
        Enhanced from doppler_physics_v2.py with configurable gap bridging.

        Args:
            clean_mask: Binary mask of valid ridge pixels
            min_length: Minimum skeleton length (default: self.min_skeleton_length)

        Returns:
            List of coordinate arrays, each (N, 2) with (freq_idx, time_idx) pairs sorted by time.
        """
        if not SKIMAGE_AVAILABLE:
            logger.warning("skimage not available for morphology")
            return []

        if min_length is None:
            min_length = self.min_skeleton_length

        try:
            logger.info(f"Morphology input: clean_mask has {clean_mask.sum()} pixels")

            # Simple morphological operations - same as doppler_physics_v2.py
            # Use horizontal structuring element for closing/opening
            selem = make_rect(1, 5)

            # Binary closing to bridge small horizontal gaps
            mask_closed = binary_closing(clean_mask, selem)
            logger.info(f"After closing: {mask_closed.sum()} pixels")

            # Binary opening to remove noise
            mask_open = binary_opening(mask_closed, selem)
            logger.info(f"After opening: {mask_open.sum()} pixels")

            mask_final = mask_open

            # Skeletonize
            skel = skeletonize(mask_final)
            logger.info(f"Skeleton: {skel.sum()} pixels")

            # Label connected components
            labels_img = label(skel, connectivity=2)
            regions = regionprops(labels_img)

            logger.info(f"Found {len(regions)} skeleton regions")

            tracks = []
            for i, r in enumerate(regions):
                coords = r.coords  # (N, 2) array of (row, col) = (freq_idx, time_idx)
                if coords.shape[0] < min_length:
                    logger.debug(f"Region {i}: rejected ({coords.shape[0]} pixels < {min_length})")
                    continue
                # Sort by time (column index)
                coords_sorted = coords[np.argsort(coords[:, 1])]
                tracks.append(coords_sorted)
                logger.debug(f"Region {i}: ACCEPTED ({coords.shape[0]} pixels)")

            return tracks

        except Exception as e:
            logger.error(f"Morphology extraction failed: {e}")
            return []

    def _coords_to_world(self, coords: np.ndarray,
                         times: np.ndarray, freqs: np.ndarray) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates to world coordinates (time, freq).

        Args:
            coords: (N, 2) array of (freq_idx, time_idx)
            times: time array for the region
            freqs: frequency array for the region

        Returns:
            List of (time, freq) tuples
        """
        centerline = []
        skipped = 0
        for f_idx, t_idx in coords:
            if t_idx < len(times) and f_idx < len(freqs):
                t_world = float(times[int(t_idx)])
                f_world = float(freqs[int(f_idx)])
                centerline.append((t_world, f_world))
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(f"_coords_to_world: skipped {skipped}/{len(coords)} points (out of bounds)")

        if centerline:
            t_vals = [p[0] for p in centerline]
            f_vals = [p[1] for p in centerline]
            logger.info(f"_coords_to_world: {len(centerline)} points, "
                       f"t=[{min(t_vals):.3f}, {max(t_vals):.3f}], "
                       f"f=[{min(f_vals):.1f}, {max(f_vals):.1f}]")
        else:
            logger.warning(f"_coords_to_world: no valid points! "
                          f"coords range: f_idx=[{coords[:,0].min()}, {coords[:,0].max()}], "
                          f"t_idx=[{coords[:,1].min()}, {coords[:,1].max()}], "
                          f"times len={len(times)}, freqs len={len(freqs)}")

        return centerline

    def _calculate_track_score(self, track: List[Tuple[float, float]]) -> float:
        """Calculate quality score for a track."""
        if len(track) < 2:
            return 0.0

        t_vals = np.array([p[0] for p in track])
        f_vals = np.array([p[1] for p in track])

        point_count = len(track)
        duration = t_vals[-1] - t_vals[0]

        # Calculate slope consistency
        if len(t_vals) > 2:
            dt = np.diff(t_vals)
            df = np.diff(f_vals)
            dt = np.where(dt == 0, 1e-6, dt)
            slopes = df / dt
            slope_std = np.std(slopes) if len(slopes) > 0 else 0
            slope_consistency = 1.0 / (1.0 + slope_std / 100.0)
        else:
            slope_consistency = 0.5

        # Combine factors
        score = (
            0.4 * min(point_count / 50.0, 1.0) +
            0.4 * min(duration / 2.0, 1.0) +
            0.2 * slope_consistency
        )

        return score

    def _smooth_track(self, centerline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply Savitzky-Golay smoothing to track."""
        if not SCIPY_AVAILABLE or len(centerline) < self.savgol_window:
            return centerline

        t_vals = np.array([p[0] for p in centerline])
        f_vals = np.array([p[1] for p in centerline])

        try:
            # Adjust window size if needed
            window = min(self.savgol_window, len(f_vals))
            if window % 2 == 0:
                window -= 1
            if window < 3:
                return centerline

            f_smooth = savgol_filter(f_vals, window, min(self.savgol_poly, window - 1))
            return [(t, f) for t, f in zip(t_vals, f_smooth)]
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}")
            return centerline

    def _get_track_endpoints(self, track: DetectedTrack) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Get track start/end points and slope.

        Returns:
            (start_point, end_point, slope) where points are (time, freq) tuples
        """
        points = track.points
        if len(points) < 2:
            return (points[0], points[0], 0.0) if points else ((0, 0), (0, 0), 0.0)

        # Start and end points
        start = points[0]
        end = points[-1]

        # Calculate average slope (Hz/s)
        t_vals = np.array([p[0] for p in points])
        f_vals = np.array([p[1] for p in points])

        dt = t_vals[-1] - t_vals[0]
        if dt > 0:
            slope = (f_vals[-1] - f_vals[0]) / dt
        else:
            slope = 0.0

        return (start, end, slope)

    def _can_bridge_tracks(self, track1: DetectedTrack, track2: DetectedTrack) -> Tuple[bool, float]:
        """
        Check if two tracks can be bridged (connected).

        Track1 should end before track2 starts.

        Returns:
            (can_bridge, gap_score) - gap_score is lower for better matches
        """
        start1, end1, slope1 = self._get_track_endpoints(track1)
        start2, end2, slope2 = self._get_track_endpoints(track2)

        # Check time ordering: track1 should end before track2 starts
        t_gap = start2[0] - end1[0]

        # Must have positive gap (track1 ends before track2 starts)
        if t_gap < 0:
            return False, float('inf')

        # Check gap is within threshold
        if t_gap > self.gap_threshold_sec:
            return False, float('inf')

        # Check frequency difference at gap
        # Extrapolate track1's end frequency to track2's start time
        expected_freq = end1[1] + slope1 * t_gap
        actual_freq = start2[1]
        freq_diff = abs(expected_freq - actual_freq)

        if freq_diff > self.gap_freq_tolerance_hz:
            return False, float('inf')

        # Check slope compatibility
        slope_diff = abs(slope1 - slope2)
        if slope_diff > self.gap_slope_tolerance:
            return False, float('inf')

        # Calculate gap score (lower is better)
        gap_score = (
            0.4 * (t_gap / self.gap_threshold_sec) +
            0.4 * (freq_diff / self.gap_freq_tolerance_hz) +
            0.2 * (slope_diff / self.gap_slope_tolerance)
        )

        logger.debug(f"Can bridge: gap={t_gap:.3f}s, freq_diff={freq_diff:.1f}Hz, "
                    f"slope_diff={slope_diff:.1f}Hz/s, score={gap_score:.3f}")

        return True, gap_score

    def _merge_two_tracks(self, track1: DetectedTrack, track2: DetectedTrack) -> DetectedTrack:
        """
        Merge two tracks by interpolating across the gap.

        Track1 should end before track2 starts.
        """
        points1 = track1.points
        points2 = track2.points

        # Get endpoints
        end1 = points1[-1]
        start2 = points2[0]

        # Interpolate gap if needed
        gap_points = []
        t_gap = start2[0] - end1[0]

        if t_gap > 0.01:  # Only interpolate if gap is significant
            # Number of interpolation points based on gap size
            n_interp = max(2, int(t_gap / 0.01))  # ~100 points per second

            t_interp = np.linspace(end1[0], start2[0], n_interp + 2)[1:-1]
            f_interp = np.linspace(end1[1], start2[1], n_interp + 2)[1:-1]

            gap_points = [(float(t), float(f)) for t, f in zip(t_interp, f_interp)]

        # Combine all points
        merged_points = points1 + gap_points + points2

        # Recalculate properties
        t_vals = [p[0] for p in merged_points]
        f_vals = [p[1] for p in merged_points]

        duration = max(t_vals) - min(t_vals)
        freq_range = (min(f_vals), max(f_vals))

        # Combine merged_from lists
        merged_from = track1.merged_from + track2.merged_from

        # Recalculate score
        score = self._calculate_track_score(merged_points)

        return DetectedTrack(
            points=merged_points,
            score=score,
            duration=duration,
            freq_range=freq_range,
            point_count=len(merged_points),
            merged_from=merged_from
        )

    def _bridge_track_gaps(self, tracks: List[DetectedTrack]) -> List[DetectedTrack]:
        """
        Bridge gaps between track segments that appear to be from the same trajectory.

        This connects short breaks in tracks caused by noise or interference.
        """
        if len(tracks) < 2:
            return tracks

        # Sort tracks by start time
        sorted_tracks = sorted(tracks, key=lambda t: t.points[0][0])

        # Track which tracks have been merged
        merged = [False] * len(sorted_tracks)
        result = []

        i = 0
        while i < len(sorted_tracks):
            if merged[i]:
                i += 1
                continue

            current = sorted_tracks[i]
            merged[i] = True

            # Try to extend current track by bridging gaps
            changed = True
            while changed:
                changed = False
                best_match_idx = -1
                best_score = float('inf')

                # Look for tracks that can be bridged to current
                for j in range(i + 1, len(sorted_tracks)):
                    if merged[j]:
                        continue

                    can_bridge, score = self._can_bridge_tracks(current, sorted_tracks[j])
                    if can_bridge and score < best_score:
                        best_score = score
                        best_match_idx = j

                if best_match_idx >= 0:
                    # Merge the best match
                    logger.info(f"Bridging track {i} with track {best_match_idx} (score={best_score:.3f})")
                    current = self._merge_two_tracks(current, sorted_tracks[best_match_idx])
                    merged[best_match_idx] = True
                    changed = True

            result.append(current)
            i += 1

        # Add any remaining un-merged tracks
        for i, track in enumerate(sorted_tracks):
            if not merged[i]:
                result.append(track)

        return result

    def _merge_similar_tracks(self, tracks: List[DetectedTrack]) -> List[DetectedTrack]:
        """
        Merge tracks that are very similar (nearly parallel, same time range).

        This handles cases where a single track was split into parallel segments.
        Note: This is different from gap bridging - this merges overlapping tracks.
        """
        if len(tracks) < 2:
            return tracks

        # Calculate track properties
        track_props = []
        for track in tracks:
            points = track.points
            t_vals = np.array([p[0] for p in points])
            f_vals = np.array([p[1] for p in points])

            t_start, t_end = t_vals[0], t_vals[-1]
            f_mean = np.mean(f_vals)

            # Calculate slope
            if t_end > t_start:
                slope = (f_vals[-1] - f_vals[0]) / (t_end - t_start)
            else:
                slope = 0.0

            track_props.append({
                'track': track,
                't_start': t_start,
                't_end': t_end,
                'f_mean': f_mean,
                'slope': slope
            })

        # Find similar tracks
        merged = [False] * len(tracks)
        result = []

        for i in range(len(tracks)):
            if merged[i]:
                continue

            current = track_props[i]
            merged[i] = True

            # Find similar tracks to merge
            similar_indices = [i]

            for j in range(i + 1, len(tracks)):
                if merged[j]:
                    continue

                other = track_props[j]

                # Check time overlap
                overlap_start = max(current['t_start'], other['t_start'])
                overlap_end = min(current['t_end'], other['t_end'])

                if overlap_end <= overlap_start:
                    continue  # No overlap

                # Check frequency similarity
                freq_diff = abs(current['f_mean'] - other['f_mean'])
                if freq_diff > self.merge_freq_tolerance:
                    continue

                # Check slope similarity
                slope_diff = abs(current['slope'] - other['slope'])
                if slope_diff > self.merge_slope_tolerance:
                    continue

                # Similar enough to merge
                similar_indices.append(j)
                merged[j] = True

            if len(similar_indices) == 1:
                # No merge needed
                result.append(tracks[i])
            else:
                # Merge similar tracks - keep the one with highest score
                # (or could combine points, but keeping best avoids duplicates)
                best_idx = max(similar_indices, key=lambda idx: tracks[idx].score)
                best_track = tracks[best_idx]

                # Update merged_from to include all merged tracks
                all_merged = []
                for idx in similar_indices:
                    all_merged.extend(tracks[idx].merged_from)

                merged_track = DetectedTrack(
                    points=best_track.points,
                    score=best_track.score,
                    duration=best_track.duration,
                    freq_range=best_track.freq_range,
                    point_count=best_track.point_count,
                    merged_from=all_merged
                )
                result.append(merged_track)

                logger.info(f"Merged {len(similar_indices)} similar tracks")

        return result


def check_dependencies() -> dict:
    """Check which dependencies are available."""
    return {
        'skimage': SKIMAGE_AVAILABLE,
        'sklearn': SKLEARN_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'full_support': SKIMAGE_AVAILABLE and SKLEARN_AVAILABLE and SCIPY_AVAILABLE
    }
