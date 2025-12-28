# Debugging Curve Mode Issue

## To help debug the issue where Enter key doesn't trigger the dialog:

1. **Run the application normally**

2. **Open the console/terminal where you launched the app** - this is where logs will appear

3. **Enter curve mode** by pressing `C`
   - You should see in the logs: "Curve mode: ON (Click to add points, Right-click to clear, Enter to finish)"

4. **Click at least 2 points on the spectrogram**
   - After each click, you should see: "Added curve point: t=X.XXXs, f=XXXX.XHz, total points: N"

5. **Press Enter**
   - You should see these logs:
     - "Enter pressed: Curve completed with N points"
     - "Points copy created: [(X, Y), (X, Y)]... (showing first 2)"
     - "Calling curve completed callback..."
     - "on_curve_completed called with N points"
     - "Showing save track dialog..."

6. **If you DON'T see these logs**, then the issue is:
   - Enter key event is not being captured
   - Curve mode is not actually active
   - Canvas doesn't have focus

## What to check in the logs:

### If you see this:
```
Enter pressed: Curve completed with N points
No curve completed callback set!
```
**Problem**: The callback wasn't registered properly.

### If you see nothing when pressing Enter:
**Problem**: The key event isn't reaching the canvas or curve_mode is False.

### If you see the logs but no dialog:
```
on_curve_completed called with N points
Showing save track dialog...
Error showing dialog: ...
```
**Problem**: There's an exception in showing the dialog.

## Quick Test

You can also run the test script to verify the basic functionality:

```bash
cd /c/Users/koren/Projects/PythonProject
python test_track_saving.py
```

This will test the track manager independently of the UI.

## Common Issues

1. **Canvas doesn't have focus**: Click on the spectrogram canvas first
2. **Enter key captured by another widget**: Make sure no text fields are focused
3. **Curve mode not active**: Press C first and verify the status message shows "CURVE MODE"
4. **Not enough points**: You need at least 2 points (status bar will tell you)

## Expected Behavior

After pressing Enter with 2+ points:
1. Dialog appears asking "Do you want to save this track?"
2. Click Yes → Another dialog asks for track label
3. Click OK → Track is saved, success message appears
4. Curve is cleared and you exit curve mode

## If Right-Click Issue

Right-click CLEARS the curve (by design). This is intentional so you can start over.
- After right-click, status should show: "CURVE MODE: Curve cleared | Click to add points..."
- Just click new points and try again
