# Troubleshooting Track Save Feature

## Current Status

I've added extensive debugging to help identify the issue. Here's what was changed:

### Changes Made:

1. **Reduced minimum points from 4 to 2** (`vispy_canvas.py:648`)
2. **Added debug print statements** that will show in the console
3. **Added try/except blocks** to catch any errors
4. **Store points copy before callback** to prevent them being cleared
5. **Improved status messages** to show progress

## How to Debug

### Step 1: Run the application from command line

Make sure you run the app from a terminal so you can see the output:

```bash
cd /c/Users/koren/Projects/PythonProject
python -m audio_visualizer.main
```

OR if you have a different launch command, use that but **make sure you can see the console output**.

### Step 2: Load an audio file and enter curve mode

1. Load an audio file
2. Press **`C`** to enter curve mode
3. You should see in the console:
   ```
   Curve mode: ON (Click to add points, Right-click to clear, Enter to finish)
   ```

### Step 3: Click points

- Click at least 2 points on the spectrogram
- After each click, check the console for:
  ```
  Added curve point: t=X.XXXs, f=XXXX.XHz, total points: N
  ```

### Step 4: Press Enter

When you press Enter, you should see these debug messages:

```
DEBUG: Enter key pressed! curve_mode=True, points=N
DEBUG: Entering save flow with N points
Enter pressed: Curve completed with N points
Points copy created: [(X, Y), (X, Y)]... (showing first 2)
Calling curve completed callback...
on_curve_completed called with N points
Showing save track dialog...
```

## What Different Outputs Mean

### If you see: "DEBUG: Enter key pressed! curve_mode=False, points=N"
**Problem**: Curve mode got turned off somehow.
**Solution**: Press `C` again before pressing Enter.

### If you see: "DEBUG: Enter key pressed! curve_mode=True, points=1"
**Problem**: Not enough points (need minimum 2).
**Solution**: Click one more point before pressing Enter.

### If you see nothing at all when pressing Enter
**Problem**: The canvas doesn't have keyboard focus.
**Solution**:
- Click on the spectrogram canvas first
- Make sure no text fields are focused
- Try clicking on the canvas, then press `C`, then click points, then press Enter

### If you see: "No curve completed callback set!"
**Problem**: The callback wasn't registered.
**This would be a bug in the code** - let me know if you see this.

### If you see: "Error showing dialog: ..."
**Problem**: There's an exception when showing the QMessageBox.
**This needs investigation** - copy the full error message.

## Right-Click Behavior (NOT A BUG)

Right-click **intentionally** clears the curve so you can start over:
- Status bar should show: "CURVE MODE: Curve cleared | Click to add points..."
- This is by design - just click new points and try again

## Alternative: Test Without UI

You can test the core functionality without the UI:

```bash
python test_track_saving.py
```

This will verify that the TrackManager, interpolation, and CSV export all work correctly.

## What to Report

If it still doesn't work, please provide:

1. **Console output** when you press Enter (copy everything that appears)
2. **Which debug message is the last one you see**
3. **Any error messages or exceptions**
4. **Whether you see "DEBUG: Enter key pressed!" at all**

This will help me identify exactly where the issue is!

## Known Working Flow

This is what SHOULD happen:

1. Press `C` → See "CURVE MODE: Click to add points..."
2. Click point 1 → See "CURVE MODE: 1 points (need 1 more)"
3. Click point 2 → See "CURVE MODE: 2 points | Click to add | Right-click to clear | Enter to finish"
4. Press `Enter` → See debug messages and a dialog asking "Do you want to save this track?"
5. Click `Yes` → Dialog asks for track label
6. Click `OK` → Success message with track info
7. Check the `<filename>_tracks/` directory for the created files

## Common Pitfalls

1. **Pressing Enter too soon**: Need at least 2 points
2. **Canvas not focused**: Click on the canvas before pressing keys
3. **Right-click clears the curve**: This is intentional, not a bug
4. **Running without console**: Can't see debug output if not run from terminal
