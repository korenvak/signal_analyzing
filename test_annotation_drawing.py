import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import numpy as np

# Since we can't be sure PySide6 is fully initialized for GUI tests,
# we will mock the QTimer to avoid issues.
with patch('PySide6.QtCore.QTimer', MagicMock()):
    from audio_visualizer.ui.vispy_canvas import VisPyCanvas
    from audio_visualizer.ui.annotation_renderer import AnnotationRenderer
    from audio_visualizer.ui.annotation_data import Annotation

class TestAnnotationDrawing(unittest.TestCase):

    def setUp(self):
        """Set up the canvas and mock renderer for each test."""
        # Mock VisPy scene and Qt timers to avoid GUI instantiation
        with patch('vispy.scene.SceneCanvas.__init__', MagicMock(return_value=None)):
             with patch('PySide6.QtCore.QTimer', MagicMock()):
                self.canvas = VisPyCanvas(view_type='spectrogram')

        # Mock the renderer
        self.renderer = MagicMock(spec=AnnotationRenderer)
        self.canvas.set_annotation_renderer(self.renderer)

        # Mock the transform logic for predictable coordinate conversion
        self.mock_transform = MagicMock()
        self.mock_transform.imap.side_effect = lambda pos: (pos[0], pos[1], 0, 0)
        
        mock_node_transform = MagicMock()
        mock_node_transform.return_value = self.mock_transform
        self.canvas.view.scene.node_transform = mock_node_transform

        # Enable annotation mode
        self.canvas.set_annotation_mode(True)

    def test_annotation_drawing_flow(self):
        """Test the full mouse drag sequence for creating an annotation."""
        # 1. Mouse Press (start drawing)
        press_event = MagicMock()
        press_event.button = 1
        press_event.pos = (100, 50)

        self.canvas.on_mouse_press(press_event)

        # Verify drawing has started
        self.assertTrue(self.canvas.annotation_drawing)
        self.assertEqual(self.canvas.annotation_start, (100, 50))

        # 2. Mouse Move (live drawing)
        move_event = MagicMock()
        move_event.pos = (200, 150)

        self.canvas.on_mouse_move(move_event)

        # Verify the temporary rectangle is shown
        self.renderer.show_temp_rectangle.assert_called_once_with(100, 200, 50, 150)

        # 3. Mouse Release (finalize annotation)
        release_event = MagicMock()
        release_event.button = 1
        release_event.pos = (200, 150)

        creation_callback = MagicMock()
        self.canvas.set_annotation_callbacks(on_created=creation_callback)
        self.canvas.on_mouse_release(release_event)

        # Verify drawing has stopped and temp rectangle is hidden
        self.assertFalse(self.canvas.annotation_drawing)
        self.renderer.hide_temp_rectangle.assert_called_once()

        # Verify the final annotation was created with correct coordinates
        creation_callback.assert_called_once_with(100, 200, 50, 150)

    def test_screen_to_world_uses_inverse_transform(self):
        """Test that _screen_to_world uses the inverse transform (imap)."""
        screen_pos = (150, 250)
        
        # The side effect in setUp already mocks the behavior of imap
        self.mock_transform.imap.return_value = (1.23, 456.78, 0, 0)

        world_pos = self.canvas._screen_to_world(screen_pos)

        # Verify that imap was called, not map
        self.mock_transform.imap.assert_called_with((150, 250, 0))
        self.mock_transform.map.assert_not_called()
        self.assertEqual(world_pos, (1.23, 456.78))

if __name__ == '__main__':
    unittest.main()