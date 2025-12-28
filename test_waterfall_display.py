"""
Quick test script for waterfall canvas display.
Creates synthetic data and displays it directly.
"""

import sys
import numpy as np

# Add project to path
sys.path.insert(0, '.')

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from PySide6.QtCore import Qt


def generate_test_data(n_time=500, n_sensors=200):
    """Generate synthetic DAS-like data with visible patterns."""
    print(f"Generating test data: {n_time} time samples x {n_sensors} sensors")

    # Create base noise
    data = np.random.randn(n_time, n_sensors).astype(np.float32) * 0.1

    # Add traveling waves (diagonal stripes)
    for i in range(3):
        speed = 2 + i  # Different speeds
        amplitude = 0.3 + i * 0.2
        freq = 0.05 + i * 0.02
        for t in range(n_time):
            phase = t / speed
            wave = amplitude * np.sin(2 * np.pi * freq * (np.arange(n_sensors) - phase))
            data[t, :] += wave

    # Add some horizontal lines (events across all sensors)
    for event_time in [100, 250, 400]:
        if event_time < n_time:
            data[event_time:event_time+5, :] += 0.5

    # Add vertical stripes (sensor anomalies)
    for sensor in [50, 100, 150]:
        if sensor < n_sensors:
            data[:, sensor] += 0.3

    # Add a bright spot
    cx, cy = n_sensors // 2, n_time // 2
    for dy in range(-20, 21):
        for dx in range(-20, 21):
            if 0 <= cy + dy < n_time and 0 <= cx + dx < n_sensors:
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 20:
                    data[cy + dy, cx + dx] += 0.5 * (1 - dist / 20)

    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Data shape: {data.shape}")

    return data


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waterfall Canvas Test")
        self.setGeometry(100, 100, 1000, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel("Click 'Load Data' to test the waterfall display")
        self.info_label.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.info_label)

        # Button row
        btn_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)
        btn_layout.addWidget(self.load_btn)

        self.reload_btn = QPushButton("Reload (New Data)")
        self.reload_btn.clicked.connect(self.reload_data)
        btn_layout.addWidget(self.reload_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_data)
        btn_layout.addWidget(self.clear_btn)

        layout.addLayout(btn_layout)

        # Create waterfall canvas
        try:
            from audio_visualizer.ui.waterfall_canvas import WaterfallCanvas
            self.canvas = WaterfallCanvas()
            self.canvas.on_cursor_moved_callback = self.on_cursor_moved
            layout.addWidget(self.canvas.native)
            print("WaterfallCanvas created successfully")
        except Exception as e:
            print(f"Failed to create WaterfallCanvas: {e}")
            import traceback
            traceback.print_exc()
            self.canvas = None
            error_label = QLabel(f"Error: {e}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)

        # Cursor info label
        self.cursor_label = QLabel("Cursor: -")
        self.cursor_label.setStyleSheet("font-family: monospace; padding: 5px; background: #222; color: #aaa;")
        layout.addWidget(self.cursor_label)

    def load_data(self):
        if self.canvas is None:
            return

        try:
            # Generate test data
            data = generate_test_data(500, 200)

            # Set data on canvas
            self.canvas.set_data(
                data=data,
                sensor_start=0,
                sensor_end=200,
                time_start=0,
                time_end=500
            )

            self.info_label.setText(f"Loaded: {data.shape[0]} x {data.shape[1]} | Range: [{data.min():.3f}, {data.max():.3f}]")
            print("Data loaded successfully!")

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"Error: {e}")

    def reload_data(self):
        """Reload with fresh random data."""
        self.load_data()

    def clear_data(self):
        if self.canvas:
            self.canvas.clear()
            self.info_label.setText("Cleared")

    def on_cursor_moved(self, info):
        """Handle cursor movement."""
        sensor = info.get('sensor', 0)
        time_idx = info.get('time_idx', 0)
        value = info.get('value')

        if value is not None:
            self.cursor_label.setText(f"Sensor: {sensor}  |  Time: {time_idx}  |  Value: {value:.4f}")
        else:
            self.cursor_label.setText(f"Sensor: {sensor}  |  Time: {time_idx}  |  Value: -")


def main():
    print("=" * 60)
    print("WATERFALL CANVAS TEST")
    print("=" * 60)

    app = QApplication(sys.argv)

    # Dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #1a1a2e;
            color: #eee;
        }
        QPushButton {
            background-color: #16213e;
            border: 1px solid #0f3460;
            padding: 8px 16px;
            color: #eee;
        }
        QPushButton:hover {
            background-color: #0f3460;
        }
    """)

    window = TestWindow()
    window.show()

    # Auto-load data on start
    window.load_data()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
