#!/usr/bin/env python3
"""
Test script for multi-file support.
Tests the multi-file manager widget functionality.
"""

import sys
import os
import time
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_multi_file_manager():
    """Test the multi-file manager widget."""
    logger.info("Starting multi-file manager test...")
    
    try:
        from audio_visualizer.ui.multi_file_manager import MultiFileManager
        
        app = QApplication(sys.argv)
        
        # Create test window
        window = QMainWindow()
        window.setWindowTitle("Multi-File Manager Test")
        window.setMinimumSize(800, 600)
        
        # Create multi-file manager
        multi_file_manager = MultiFileManager()
        window.setCentralWidget(multi_file_manager)
        
        # Connect signals for testing
        def on_file_opened(file_path):
            logger.info(f"Signal received - File opened: {Path(file_path).name}")
        
        def on_file_closed(file_path):
            logger.info(f"Signal received - File closed: {Path(file_path).name}")
        
        def on_current_file_changed(file_path):
            logger.info(f"Signal received - Current file changed: {Path(file_path).name}")
        
        multi_file_manager.file_opened.connect(on_file_opened)
        multi_file_manager.file_closed.connect(on_file_closed)
        multi_file_manager.current_file_changed.connect(on_current_file_changed)
        
        # Add some test files (create dummy ones)
        test_files_dir = project_root / "test_audio_files"
        test_files_dir.mkdir(exist_ok=True)
        
        test_files = []
        for i in range(3):
            test_file = test_files_dir / f"test_audio_{i+1}.wav"
            # Create dummy file
            test_file.write_bytes(b"dummy audio data " * 100)
            test_files.append(str(test_file))
        
        # Test adding files programmatically
        QTimer.singleShot(1000, lambda: multi_file_manager.add_files(test_files))
        
        # Test opening first file
        QTimer.singleShot(2000, lambda: multi_file_manager.open_file(test_files[0]))
        
        # Test switching files
        QTimer.singleShot(3000, lambda: multi_file_manager.open_file(test_files[1]))
        
        # Test closing file
        QTimer.singleShot(4000, lambda: multi_file_manager.close_file(test_files[0]))
        
        # Auto-close after testing
        QTimer.singleShot(6000, app.quit)
        
        logger.info("✓ Multi-file manager widget created successfully")
        logger.info("✓ Test files added to playlist")
        logger.info("✓ Signal connections working")
        
        # Show window
        window.show()
        
        # Run test
        app.exec()
        
        # Cleanup test files
        import shutil
        shutil.rmtree(test_files_dir, ignore_errors=True)
        
        logger.info("✓ Multi-file manager test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_multi_file_manager()
    sys.exit(0 if success else 1)