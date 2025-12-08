"""
Performance Optimizer
Implements adaptive quality, movement detection, and smart throttling
for maximum performance during zoom/pan operations.
"""

import time
import threading
from typing import Tuple, Optional, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MovementDetector:
    """Detects fast movement for adaptive quality reduction."""
    
    def __init__(self, fast_threshold: float = 0.1, history_size: int = 5):
        """Initialize movement detector.
        
        Args:
            fast_threshold: Time threshold in seconds - if viewport changes faster than this, it's "fast"
            history_size: Number of recent viewport changes to track
        """
        self.fast_threshold = fast_threshold
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=history_size)  # (timestamp, time_range_span, freq_range_span)
        self.lock = threading.Lock()
        self.is_moving = False
        self.last_movement_time = 0.0
    
    def update_viewport(self, time_range: Tuple[float, float], freq_range: Tuple[float, float]):
        """Update viewport and detect movement speed.
        
        Returns:
            True if movement is fast, False otherwise
        """
        current_time = time.time()
        time_span = time_range[1] - time_range[0]
        freq_span = freq_range[1] - freq_range[0]
        
        with self.lock:
            if len(self.history) > 0:
                last_time, last_time_span, last_freq_span = self.history[-1]
                time_since_last = current_time - last_time
                
                # Calculate change rate
                time_change_rate = abs(time_span - last_time_span) / max(time_span, last_time_span, 1e-9)
                freq_change_rate = abs(freq_span - last_freq_span) / max(freq_span, last_freq_span, 1e-9)
                total_change_rate = (time_change_rate + freq_change_rate) / 2.0
                
                # Fast movement if change is significant and happened quickly
                is_fast = time_since_last < self.fast_threshold and total_change_rate > 0.05
                
                if is_fast:
                    self.is_moving = True
                    self.last_movement_time = current_time
                elif current_time - self.last_movement_time > 0.3:  # 300ms of no fast movement
                    self.is_moving = False
            else:
                self.is_moving = False
            
            self.history.append((current_time, time_span, freq_span))
        
        return self.is_moving
    
    def is_fast_movement(self) -> bool:
        """Check if currently experiencing fast movement."""
        with self.lock:
            # Also check if movement was recent (within last 300ms)
            if self.is_moving and time.time() - self.last_movement_time < 0.3:
                return True
            return False


class AdaptiveQualityManager:
    """Manages adaptive quality reduction during fast movement."""
    
    def __init__(self):
        """Initialize adaptive quality manager."""
        self.movement_detector = MovementDetector()
        self.quality_reduction_active = False
        self.base_lod_offset = 0  # Additional LOD offset during fast movement
        self.refinement_timer = None
        self.refinement_pending = False
        self.lock = threading.Lock()
    
    def get_effective_lod(self, base_lod: int, zoom_level: float) -> int:
        """Get effective LOD level considering adaptive quality.
        
        Args:
            base_lod: Base LOD level based on zoom
            zoom_level: Current zoom level
        
        Returns:
            Effective LOD (may be higher = lower quality during fast movement)
        """
        with self.lock:
            if self.quality_reduction_active:
                # Add 1-2 LOD levels during fast movement for faster computation
                effective_lod = min(base_lod + self.base_lod_offset + 1, 5)
                return effective_lod
            return base_lod
    
    def update_viewport(self, time_range: Tuple[float, float], freq_range: Tuple[float, float]):
        """Update viewport and adjust quality accordingly.
        
        Returns:
            (is_fast_movement, effective_lod_offset)
        """
        is_fast = self.movement_detector.update_viewport(time_range, freq_range)
        
        with self.lock:
            if is_fast and not self.quality_reduction_active:
                # Start quality reduction
                self.quality_reduction_active = True
                self.base_lod_offset = 1  # Use lower quality during fast movement
                logger.debug("Fast movement detected - reducing quality for performance")
            elif not is_fast and self.quality_reduction_active:
                # Schedule refinement when movement stops
                self._schedule_refinement()
        
        return is_fast, self.base_lod_offset if self.quality_reduction_active else 0
    
    def _schedule_refinement(self):
        """Schedule quality refinement after movement stops."""
        if self.refinement_timer is not None:
            return  # Already scheduled
        
        def do_refinement():
            time.sleep(0.2)  # Wait 200ms after movement stops
            with self.lock:
                if not self.movement_detector.is_fast_movement():
                    self.quality_reduction_active = False
                    self.base_lod_offset = 0
                    self.refinement_pending = True
                    logger.debug("Movement stopped - scheduling quality refinement")
                self.refinement_timer = None
        
        self.refinement_timer = threading.Thread(target=do_refinement, daemon=True)
        self.refinement_timer.start()
    
    def should_refine(self) -> bool:
        """Check if quality refinement should happen."""
        with self.lock:
            if self.refinement_pending:
                self.refinement_pending = False
                return True
            return False


class ViewportThrottler:
    """Throttles viewport updates to reduce computation load."""
    
    def __init__(self, min_interval: float = 0.05):
        """Initialize viewport throttler.
        
        Args:
            min_interval: Minimum time between viewport updates (seconds)
        """
        self.min_interval = min_interval
        self.last_update_time = 0.0
        self.pending_update = None
        self.lock = threading.Lock()
    
    def should_update(self) -> bool:
        """Check if viewport update should be processed.
        
        Returns:
            True if update should proceed, False if should be throttled
        """
        current_time = time.time()
        with self.lock:
            if current_time - self.last_update_time >= self.min_interval:
                self.last_update_time = current_time
                return True
            return False
    
    def reset(self):
        """Reset throttler (e.g., when movement stops)."""
        with self.lock:
            self.last_update_time = 0.0


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.adaptive_quality = AdaptiveQualityManager()
        self.viewport_throttler = ViewportThrottler(min_interval=0.05)  # Max 20 updates/sec
        self.stats = {
            'fast_movement_detected': 0,
            'quality_reductions': 0,
            'throttled_updates': 0,
            'refinements': 0
        }
    
    def update_viewport(self, time_range: Tuple[float, float], 
                       freq_range: Tuple[float, float]) -> Tuple[bool, int]:
        """Update viewport with performance optimizations.
        
        Args:
            time_range: Current time range
            freq_range: Current frequency range
        
        Returns:
            (should_update, lod_offset) - whether to process update and LOD offset to apply
        """
        # Check throttling
        if not self.viewport_throttler.should_update():
            self.stats['throttled_updates'] += 1
            return False, 0
        
        # Check for fast movement and adjust quality
        is_fast, lod_offset = self.adaptive_quality.update_viewport(time_range, freq_range)
        
        if is_fast:
            self.stats['fast_movement_detected'] += 1
            if lod_offset > 0:
                self.stats['quality_reductions'] += 1
        
        return True, lod_offset
    
    def get_effective_lod(self, base_lod: int, zoom_level: float) -> int:
        """Get effective LOD with adaptive quality adjustments."""
        return self.adaptive_quality.get_effective_lod(base_lod, zoom_level)
    
    def should_refine_quality(self) -> bool:
        """Check if quality should be refined (movement stopped)."""
        if self.adaptive_quality.should_refine():
            self.stats['refinements'] += 1
            return True
        return False
    
    def reset(self):
        """Reset optimizer state."""
        self.viewport_throttler.reset()
        self.adaptive_quality.quality_reduction_active = False
        self.adaptive_quality.base_lod_offset = 0
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return self.stats.copy()

