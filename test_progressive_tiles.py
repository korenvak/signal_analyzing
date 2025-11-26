#!/usr/bin/env python3
"""
Test script for progressive tile loading integration.
Tests that the TileManager properly integrates with ProgressiveTileLoader.
"""

import sys
import os
import time
import logging
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from audio_visualizer.core.tile_cache import TileCache
from audio_visualizer.core.tile_manager import TileManager as TileMgr
from audio_visualizer.core.progressive_tile_loader import ProgressiveTileLoader
from audio_visualizer.engines.spectrogram_engine import SpectrogramEngine
from audio_visualizer.engines.cepstrogram_engine import CepstrogramEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_progressive_tile_integration():
    """Test progressive tile loading integration."""
    logger.info("Starting progressive tile integration test...")
    
    try:
        # Create cache directory
        cache_dir = project_root / "test_cache_progressive"
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        tile_cache = TileCache(
            cache_dir=str(cache_dir),
            max_memory_tiles=100,
            max_disk_gb=1.0,
            enable_compression=True
        )
        
        # Create mock cache and task managers for engines
        from audio_visualizer.core.cache_manager import CacheManager
        from audio_visualizer.core.task_manager import TaskManager
        
        cache_manager = CacheManager(max_memory_mb=512, max_gpu_memory_mb=256)
        task_manager = TaskManager()
        
        # Create engines with proper initialization
        spectrogram_engine = SpectrogramEngine(cache_manager, task_manager)
        cepstrogram_engine = CepstrogramEngine(cache_manager, task_manager, spectrogram_engine)
        
        # Add some test audio data to engines
        test_audio = np.random.randn(44100 * 5).astype(np.float32)  # 5 seconds
        spectrogram_engine.audio_data = test_audio
        cepstrogram_engine.audio_data = test_audio
        
        engines = {
            'spectrogram': spectrogram_engine,
            'cepstrogram': cepstrogram_engine
        }
        
        # Create tile manager (this should initialize progressive loader)
        tile_manager = TileMgr(tile_cache=tile_cache, engines=engines)
        
        logger.info("✓ TileManager initialized successfully")
        
        # Test that progressive loader was created
        assert hasattr(tile_manager, 'progressive_loader'), "Progressive loader not initialized"
        assert tile_manager.progressive_loader is not None, "Progressive loader is None"
        logger.info("✓ Progressive loader initialized")
        
        # Test viewport update (should trigger tile requests)
        time_range = (0.0, 10.0)  # 10 seconds
        freq_range = (0.0, 22050.0)  # Full frequency range
        zoom_level = 1.0
        
        logger.info("Testing viewport update...")
        tile_manager.update_visible_region('spectrogram', time_range, freq_range, zoom_level)
        
        # Wait a moment for background processing
        time.sleep(2.0)
        
        # Check statistics
        stats = tile_manager.get_stats()
        logger.info(f"Tile manager stats: {stats}")
        
        progressive_stats = stats.get('progressive_loader', {})
        if progressive_stats:
            logger.info(f"Progressive loader stats: {progressive_stats}")
            logger.info(f"✓ Tiles requested: {progressive_stats.get('tiles_requested', 0)}")
            logger.info(f"✓ Active requests: {progressive_stats.get('active_requests', 0)}")
            logger.info(f"✓ Cache hit rate: {progressive_stats.get('cache_hit_rate', 0):.1f}%")
        
        # Test atlas statistics
        atlas_stats = stats.get('atlases', {})
        for view_type, atlas_stat in atlas_stats.items():
            logger.info(f"Atlas {view_type}: {atlas_stat['slots_used']}/{atlas_stat['max_tiles']} slots used")
        
        # Test different viewport (should trigger different tiles)
        logger.info("Testing different viewport...")
        tile_manager.update_visible_region('cepstrogram', (5.0, 15.0), (0.0, 4000.0), 2.0)
        
        # Wait for processing
        time.sleep(1.0)
        
        # Final statistics
        final_stats = tile_manager.get_stats()
        final_progressive_stats = final_stats.get('progressive_loader', {})
        if final_progressive_stats:
            logger.info(f"Final progressive stats: {final_progressive_stats}")
        
        # Shutdown properly
        logger.info("Shutting down tile manager...")
        tile_manager.shutdown()
        
        logger.info("✓ Progressive tile integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_progressive_tile_integration()
    sys.exit(0 if success else 1)