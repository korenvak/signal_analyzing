"""
Full spectrogram Doppler track detector.

Detects Doppler tracks across the entire spectrogram using peak detection
and track linking. Works on the already-filtered spectrogram data.

Based on doppler_detector's auto_detector.py approach.
"""
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DopplerTrack:
    """A detected Doppler track."""
    points: List[Tuple[int, int]]  # (time_idx, freq_idx) pairs
    times: List[float]  # Time values in seconds
    freqs: List[float]  # Frequency values in Hz
    avg_power: float  # Average power along track
    duration: float  # Duration in seconds
    freq_range: Tuple[float, float]  # (min_freq, max_freq)

    def __repr__(self):
        return (f"DopplerTrack(points={len(self.points)}, duration={self.duration:.2f}s, "
                f"freq={self.freq_range[0]:.0f}-{self.freq_range[1]:.0f}Hz, "
                f"avg_power={self.avg_power:.3f})")


class SpectrogramDetector:
    """
    Detects Doppler tracks in the full spectrogram using peak-based detection.

    Pipeline:
    1. For each time frame, find peaks in the frequency spectrum
    2. Link peaks frame-to-frame into continuous tracks
    3. Filter tracks by duration, point count, power
    4. Merge nearby fragmented tracks
    """

    def __init__(self,
                 # Frequency range
                 freq_min: float = 50.0,
                 freq_max: float = 1500.0,
                 # Peak detection thresholds (adaptive)
                 power_threshold_mode: str = 'adaptive',  # 'adaptive' or 'fixed'
                 power_threshold_fixed: float = 0.2,
                 power_threshold_percentile: float = 75.0,  # For adaptive mode
                 peak_prominence: float = 0.05,
                 # Track linking
                 max_gap_frames: int = 4,
                 gap_power_factor: float = 0.8,
                 gap_prominence_factor: float = 0.8,
                 max_freq_jump_hz: float = 50.0,
                 gap_max_jump_hz: float = 30.0,
                 max_peaks_per_frame: int = 20,
                 # Track filtering
                 min_track_length_frames: int = 10,
                 min_track_avg_power: float = 0.1,
                 max_track_freq_std_hz: float = 200.0,
                 # Merging
                 merge_gap_frames: int = 50,
                 merge_max_freq_diff_hz: float = 50.0,
                 enable_post_merge: bool = True):
        """
        Initialize detector parameters.

        Args:
            freq_min: Minimum frequency to consider (Hz)
            freq_max: Maximum frequency to consider (Hz)
            power_threshold_mode: 'adaptive' (based on percentile) or 'fixed'
            power_threshold_fixed: Fixed power threshold (0-1)
            power_threshold_percentile: Percentile for adaptive threshold
            peak_prominence: Minimum prominence for peaks
            max_gap_frames: Max frames to skip when linking tracks
            gap_power_factor: Relaxed power threshold for gap bridging
            gap_prominence_factor: Relaxed prominence for gap bridging
            max_freq_jump_hz: Max frequency jump between frames (Hz)
            gap_max_jump_hz: Max freq jump during gap bridging (Hz)
            max_peaks_per_frame: Maximum peaks to consider per frame
            min_track_length_frames: Minimum track length (frames)
            min_track_avg_power: Minimum average power for valid track
            max_track_freq_std_hz: Maximum freq standard deviation (Hz)
            merge_gap_frames: Max gap for merging fragmented tracks
            merge_max_freq_diff_hz: Max freq difference for merging
            enable_post_merge: Whether to merge tracks after detection
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.power_threshold_mode = power_threshold_mode
        self.power_threshold_fixed = power_threshold_fixed
        self.power_threshold_percentile = power_threshold_percentile
        self.peak_prominence = peak_prominence
        self.max_gap_frames = max_gap_frames
        self.gap_power_factor = gap_power_factor
        self.gap_prominence_factor = gap_prominence_factor
        self.max_freq_jump_hz = max_freq_jump_hz
        self.gap_max_jump_hz = gap_max_jump_hz
        self.max_peaks_per_frame = max_peaks_per_frame
        self.min_track_length_frames = min_track_length_frames
        self.min_track_avg_power = min_track_avg_power
        self.max_track_freq_std_hz = max_track_freq_std_hz
        self.merge_gap_frames = merge_gap_frames
        self.merge_max_freq_diff_hz = merge_max_freq_diff_hz
        self.enable_post_merge = enable_post_merge

        # Runtime state
        self.freqs = None
        self.times = None
        self.spectrogram = None

    def detect(self,
               spectrogram: np.ndarray,
               freqs: np.ndarray,
               times: np.ndarray,
               progress_callback=None) -> List[DopplerTrack]:
        """
        Detect Doppler tracks in the spectrogram.

        Args:
            spectrogram: 2D array (freq x time), already filtered
            freqs: Frequency values for rows
            times: Time values for columns
            progress_callback: Optional callback(progress, message)

        Returns:
            List of DopplerTrack objects
        """
        if not SCIPY_AVAILABLE:
            logger.error("scipy not available for detection")
            return []

        start_time = time.perf_counter()

        self.spectrogram = spectrogram
        self.freqs = freqs
        self.times = times

        n_freq, n_time = spectrogram.shape
        logger.info(f"Starting detection: shape=({n_freq}, {n_time}), "
                   f"freq_range=[{freqs[0]:.0f}, {freqs[-1]:.0f}]")

        # Calculate adaptive threshold if needed
        power_threshold = self._calculate_threshold(spectrogram)
        logger.info(f"Using power threshold: {power_threshold:.4f}")

        # Step 1: Detect peaks per frame
        if progress_callback:
            progress_callback(0.1, "Detecting peaks...")

        peaks_per_frame = self._detect_peaks_per_frame(power_threshold, progress_callback)

        total_peaks = sum(len(p) for p in peaks_per_frame)
        logger.info(f"Found {total_peaks} peaks across {n_time} frames")

        if total_peaks < self.min_track_length_frames:
            logger.info("Not enough peaks for track detection")
            return []

        # Step 2: Link peaks into tracks
        if progress_callback:
            progress_callback(0.4, "Linking tracks...")

        raw_tracks = self._link_peaks_to_tracks(peaks_per_frame, progress_callback)
        logger.info(f"Initial track linking: {len(raw_tracks)} tracks")

        if not raw_tracks:
            return []

        # Step 3: Filter tracks
        if progress_callback:
            progress_callback(0.7, "Filtering tracks...")

        filtered_tracks = self._filter_tracks(raw_tracks)
        logger.info(f"After filtering: {len(filtered_tracks)} tracks")

        # Step 4: Merge fragmented tracks
        if self.enable_post_merge and len(filtered_tracks) > 1:
            if progress_callback:
                progress_callback(0.85, "Merging tracks...")

            merged_tracks = self._merge_tracks(filtered_tracks)
            logger.info(f"After merging: {len(merged_tracks)} tracks")
        else:
            merged_tracks = filtered_tracks

        # Step 5: Convert to DopplerTrack objects
        if progress_callback:
            progress_callback(0.95, "Finalizing tracks...")

        result = self._convert_to_doppler_tracks(merged_tracks)

        elapsed = time.perf_counter() - start_time
        logger.info(f"Detection complete: {len(result)} tracks in {elapsed:.2f}s")

        if progress_callback:
            progress_callback(1.0, f"Found {len(result)} tracks")

        return result

    def _calculate_threshold(self, spectrogram: np.ndarray) -> float:
        """Calculate power threshold based on mode."""
        if self.power_threshold_mode == 'fixed':
            return self.power_threshold_fixed

        # Adaptive mode: use percentile of the spectrogram
        # Focus on the frequency range of interest
        f_mask = (self.freqs >= self.freq_min) & (self.freqs <= self.freq_max)
        if np.any(f_mask):
            region = spectrogram[f_mask, :]
        else:
            region = spectrogram

        # Use median-based threshold for robustness
        median_val = np.median(region)
        std_val = np.std(region)

        # Threshold at median + some factor of std
        # Or use percentile directly
        threshold = np.percentile(region, self.power_threshold_percentile)

        # Ensure it's not too low or too high
        threshold = max(threshold, median_val + 0.5 * std_val)
        threshold = min(threshold, np.percentile(region, 95))

        return threshold

    def _detect_peaks_per_frame(self, power_threshold: float,
                                progress_callback=None) -> List[List[int]]:
        """Find peaks in each time frame."""
        f = self.freqs
        S = self.spectrogram

        # Find frequency index bounds
        i_min = np.searchsorted(f, self.freq_min, side='left')
        i_max = np.searchsorted(f, self.freq_max, side='right') - 1
        i_min = max(i_min, 0)
        i_max = min(i_max, len(f) - 1)

        n_time = S.shape[1]
        peaks_per_frame = [[] for _ in range(n_time)]

        for ti in range(n_time):
            col = S[i_min:i_max + 1, ti]

            # Find peaks
            try:
                idxs, props = find_peaks(
                    col,
                    height=power_threshold,
                    prominence=self.peak_prominence
                )
            except Exception as e:
                logger.debug(f"find_peaks failed at frame {ti}: {e}")
                continue

            if idxs.size > 0:
                heights = props['peak_heights']
                # Sort by height (descending) and take top N
                order = np.argsort(heights)[::-1][:self.max_peaks_per_frame]
                # Convert to absolute frequency indices
                absolute = (idxs[order] + i_min).tolist()
                peaks_per_frame[ti] = absolute

            # Progress update
            if progress_callback and ti % 100 == 0:
                progress = 0.1 + 0.3 * (ti / n_time)
                progress_callback(progress, f"Detecting peaks... {ti}/{n_time}")

        return peaks_per_frame

    def _link_peaks_to_tracks(self, peaks_per_frame: List[List[int]],
                              progress_callback=None) -> List[List[Tuple[int, int]]]:
        """Link peaks frame-to-frame into tracks."""
        f = self.freqs
        S = self.spectrogram

        finished = []
        active = []  # Each: (last_ti, last_fi, gap_count, track_points)

        for ti, frame_peaks in enumerate(peaks_per_frame):
            used = set()
            new_active = []

            # Try to extend existing tracks
            for last_ti, last_fi, gap, track in active:
                if gap > self.max_gap_frames:
                    # Track has timed out
                    finished.append(track)
                    continue

                prev_freq = f[last_fi]
                prev_power = S[last_fi, last_ti]
                match = None
                best_dist = None

                # Strict match: find closest peak within frequency tolerance
                for pi, fi in enumerate(frame_peaks):
                    if pi in used:
                        continue

                    curr_freq = f[fi]
                    freq_diff = abs(curr_freq - prev_freq)

                    if freq_diff <= self.max_freq_jump_hz:
                        # Check power requirement
                        if S[fi, ti] >= prev_power * self.gap_power_factor:
                            if best_dist is None or freq_diff < best_dist:
                                best_dist = freq_diff
                                match = pi

                if match is not None:
                    # Found a match - extend track
                    used.add(match)
                    fi = frame_peaks[match]
                    new_active.append((ti, fi, 0, track + [(ti, fi)]))
                    continue

                # Gap bridging: look for peaks with relaxed criteria
                lower = prev_freq - self.gap_max_jump_hz
                upper = prev_freq + self.gap_max_jump_hz
                li = np.searchsorted(f, lower, side='left')
                ui = np.searchsorted(f, upper, side='right') - 1
                li, ui = max(li, 0), min(ui, len(f) - 1)

                if ui > li:
                    segment = S[li:ui+1, ti]
                    try:
                        pks, _ = find_peaks(
                            segment,
                            height=S[last_fi, last_ti] * self.gap_power_factor * 0.5,
                            prominence=self.peak_prominence * self.gap_prominence_factor
                        )
                    except:
                        pks = np.array([])

                    if pks.size > 0:
                        cand = pks + li
                        dists = np.abs(f[cand] - prev_freq)
                        best = cand[np.argmin(dists)]
                        new_active.append((ti, best, 0, track + [(ti, best)]))
                        continue

                # No match found - increment gap counter
                new_active.append((last_ti, last_fi, gap + 1, track))

            # Start new tracks from unused peaks
            for pi, fi in enumerate(frame_peaks):
                if pi not in used:
                    new_active.append((ti, fi, 0, [(ti, fi)]))

            # Update active list
            active = [t for t in new_active if t[2] <= self.max_gap_frames]
            finished.extend([t[3] for t in new_active if t[2] > self.max_gap_frames])

            # Progress
            if progress_callback and ti % 100 == 0:
                progress = 0.4 + 0.3 * (ti / len(peaks_per_frame))
                progress_callback(progress, f"Linking tracks... {ti}/{len(peaks_per_frame)}")

        # Finish remaining active tracks
        finished.extend([t[3] for t in active])

        return finished

    def _filter_tracks(self, tracks: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """Filter tracks by length, power, and frequency variance."""
        f = self.freqs
        S = self.spectrogram

        valid = []
        for track in tracks:
            # Check minimum length
            if len(track) < self.min_track_length_frames:
                continue

            # Get track properties
            freq_indices = [pt[1] for pt in track]
            powers = [S[fi, ti] for ti, fi in track]
            freqs_hz = f[freq_indices]

            # Check average power
            avg_power = np.mean(powers)
            if avg_power < self.min_track_avg_power:
                continue

            # Check frequency variance
            freq_std = np.std(freqs_hz)
            if freq_std > self.max_track_freq_std_hz:
                continue

            valid.append(track)

        return valid

    def _merge_tracks(self, tracks: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """Merge nearby fragmented tracks."""
        f = self.freqs

        # Build track info
        events = []
        for i, track in enumerate(tracks):
            tis = [pt[0] for pt in track]
            fis = [pt[1] for pt in track]
            events.append({
                'idx': i,
                'track': track,
                'start': tis[0],
                'end': tis[-1],
                'fstart': f[fis[0]],
                'fend': f[fis[-1]]
            })

        # Sort by start time
        events.sort(key=lambda e: e['start'])

        # Greedy merging
        merged = []
        for e in events:
            if not merged:
                merged.append(e)
                continue

            last = merged[-1]
            gap = e['start'] - last['end']
            freq_diff = abs(e['fstart'] - last['fend'])

            if gap <= self.merge_gap_frames and freq_diff <= self.merge_max_freq_diff_hz:
                # Merge: extend last track
                last['track'] = last['track'] + e['track']
                last['end'] = e['end']
                last['fend'] = e['fend']
            else:
                merged.append(e)

        return [e['track'] for e in merged]

    def _convert_to_doppler_tracks(self, tracks: List[List[Tuple[int, int]]]) -> List[DopplerTrack]:
        """Convert raw track data to DopplerTrack objects."""
        result = []

        for track in tracks:
            if len(track) < 2:
                continue

            time_indices = [pt[0] for pt in track]
            freq_indices = [pt[1] for pt in track]

            times_s = [self.times[ti] for ti in time_indices]
            freqs_hz = [self.freqs[fi] for fi in freq_indices]
            powers = [self.spectrogram[fi, ti] for ti, fi in track]

            doppler_track = DopplerTrack(
                points=track,
                times=times_s,
                freqs=freqs_hz,
                avg_power=float(np.mean(powers)),
                duration=times_s[-1] - times_s[0],
                freq_range=(min(freqs_hz), max(freqs_hz))
            )
            result.append(doppler_track)

        # Sort by duration (longest first)
        result.sort(key=lambda t: t.duration, reverse=True)

        return result


def check_dependencies() -> dict:
    """Check which dependencies are available."""
    return {
        'scipy': SCIPY_AVAILABLE,
        'full_support': SCIPY_AVAILABLE
    }
