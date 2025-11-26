#!/usr/bin/env python3
"""
Test script for texture atlas integration with the main UI.
Tests that atlas rendering works properly with the progressive tile system.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_atlas_integration():
    """Test texture atlas integration."""
    logger.info("Starting texture atlas integration test...")
    
    try:
        # Import atlas components
        from audio_visualizer.rendering.texture_atlas import TextureAtlas, AtlasRenderer
        from audio_visualizer.core.tile_cache import TileCache
        from audio_visualizer.core.tile_manager import TileManager as TileMgr
        from audio_visualizer.engines.spectrogram_engine import SpectrogramEngine
        from audio_visualizer.engines.cepstrogram_engine import CepstrogramEngine
        from audio_visualizer.core.cache_manager import CacheManager
        from audio_visualizer.core.task_manager import TaskManager
        
        # Create cache directory
        cache_dir = project_root / "test_cache_atlas"
        cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        tile_cache = TileCache(
            cache_dir=str(cache_dir),
            max_memory_tiles=50,
            max_disk_gb=0.5,
            enable_compression=True
        )
        
        # Create engines
        cache_manager = CacheManager(max_memory_mb=256, max_gpu_memory_mb=128)
        task_manager = TaskManager()
        
        spectrogram_engine = SpectrogramEngine(cache_manager, task_manager)
        cepstrogram_engine = CepstrogramEngine(cache_manager, task_manager, spectrogram_engine)
        
        # Add test audio data
        test_audio = np.random.randn(44100 * 10).astype(np.float32)  # 10 seconds
        spectrogram_engine.audio_data = test_audio
        cepstrogram_engine.audio_data = test_audio
        
        engines = {
            'spectrogram': spectrogram_engine,
            'cepstrogram': cepstrogram_engine
        }
        
        # Create tile manager with progressive loader
        tile_manager = TileMgr(tile_cache=tile_cache, engines=engines)
        
        logger.info("✓ Tile manager with atlas integration initialized")
        
        # Test atlas creation and access
        spectrogram_atlas = tile_manager.get_atlas('spectrogram')
        cepstrogram_atlas = tile_manager.get_atlas('cepstrogram')
        
        assert spectrogram_atlas is not None, "Spectrogram atlas not available"
        assert cepstrogram_atlas is not None, "Cepstrogram atlas not available"
        
        logger.info("✓ Atlases accessible through tile manager")
        
        # Test atlas renderer creation
        spectrogram_renderer = AtlasRenderer(spectrogram_atlas)
        cepstrogram_renderer = AtlasRenderer(cepstrogram_atlas)
        
        logger.info("✓ Atlas renderers created successfully")
        
        # Test viewport updates that should populate atlases
        time_range = (0.0, 8.0)  # 8 seconds
        freq_range = (0.0, 22050.0)  # Full frequency range
        zoom_level = 1.0
        
        logger.info("Testing spectrogram atlas population...")
        tile_manager.update_visible_region('spectrogram', time_range, freq_range, zoom_level)
        
        # Wait for background processing
        time.sleep(3.0)
        
        # Check atlas statistics
        spec_stats = spectrogram_atlas.get_stats()
        logger.info(f"Spectrogram atlas stats: {spec_stats}")
        
        if spec_stats['slots_used'] > 0:
            logger.info(f"✓ Spectrogram atlas has {spec_stats['slots_used']} tiles loaded")
            
            # Test atlas data retrieval
            atlas_data = spectrogram_atlas.get_atlas_data()
            logger.info(f"✓ Atlas data shape: {atlas_data.shape}, dtype: {atlas_data.dtype}")
            
            # Test tile mapping
            tile_mapping = spectrogram_atlas.get_tile_mapping()
            logger.info(f"✓ Tile mapping has {len(tile_mapping)} entries")
            
        else:
            logger.warning("⚠ No tiles loaded in spectrogram atlas")
        
        # Test cepstrogram atlas
        logger.info("Testing cepstrogram atlas population...")
        tile_manager.update_visible_region('cepstrogram', time_range, (0.0, 4000.0), zoom_level)
        
        time.sleep(2.0)
        
        cep_stats = cepstrogram_atlas.get_stats()
        logger.info(f"Cepstrogram atlas stats: {cep_stats}")
        
        if cep_stats['slots_used'] > 0:
            logger.info(f"✓ Cepstrogram atlas has {cep_stats['slots_used']} tiles loaded")
        
        # Test progressive loader statistics
        prog_stats = tile_manager.get_stats().get('progressive_loader', {})
        logger.info(f"Progressive loader stats: {prog_stats}")
        
        logger.info(f"✓ Total tiles computed: {prog_stats.get('tiles_computed', 0)}")
        logger.info(f"✓ Cache hit rate: {prog_stats.get('cache_hit_rate', 0):.1f}%")
        logger.info(f"✓ Average computation time: {prog_stats.get('avg_computation_time', 0):.3f}s")
        
        # Test atlas clear functionality
        logger.info("Testing atlas clear...")
        spectrogram_atlas.clear()
        cepstrogram_atlas.clear()
        
        cleared_spec_stats = spectrogram_atlas.get_stats()
        cleared_cep_stats = cepstrogram_atlas.get_stats()
        
        assert cleared_spec_stats['slots_used'] == 0, "Spectrogram atlas not properly cleared"
        assert cleared_cep_stats['slots_used'] == 0, "Cepstrogram atlas not properly cleared"
        
        logger.info("✓ Atlas clear functionality working")
        
        # Cleanup
        logger.info("Shutting down...")
        tile_manager.shutdown()
        
        logger.info("✓ Texture atlas integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_atlas_integration()
    sys.exit(0 if success else 1)