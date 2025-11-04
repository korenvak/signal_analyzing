# Audio Visualizer - Navigation Guide

## 🖱️ Mouse Controls

### Zooming

| Action | Effect | Use Case |
|--------|--------|----------|
| **Scroll Wheel** | Zoom both axes | General zoom in/out |
| **Shift + Scroll** | Zoom time axis (X) only | Focus on time resolution |
| **Ctrl + Scroll** | Zoom frequency axis (Y) only | Focus on frequency detail |

**Zoom Speed**: 2.5x per scroll (ultra-fast & instant response!)

### Panning (Navigation)

| Action | Effect |
|--------|--------|
| **Left Click + Drag** | Pan in any direction |
| **Release** | Stop panning |

**How it works**:
1. Click and hold left mouse button
2. Drag in any direction
3. View follows your mouse smoothly
4. Release to stop

**Boundaries**: Panning is constrained - you can't drag outside the spectrogram data.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **+** or **=** | Zoom in (centered) |
| **-** or **_** | Zoom out (centered) |
| **0** | Reset view to fit all data |
| **Alt** | Enable crosshair (while held) |

---

## 📊 Navigation Workflow Examples

### Example 1: Quick Explore
```
1. Load audio file
2. Scroll wheel to zoom in on interesting region
3. Drag mouse to pan to different areas
4. Ctrl+Scroll to adjust frequency range
5. Shift+Scroll to adjust time range
```

### Example 2: Detailed Analysis
```
1. Zoom in (5-10 scrolls) on specific time region
2. Drag to navigate through timeline
3. Ctrl+Scroll to zoom frequency axis for detail
4. Hold Alt to enable crosshair for precise readings
5. Drag to next region without zooming out!
```

### Example 3: Compare Two Regions
```
1. Zoom in on Region A (e.g., 100-200s)
2. Analyze features
3. Drag left/right to Region B (e.g., 800-900s)
4. Compare without re-zooming!
5. Drag back to Region A as needed
```

---

## 🎯 Pro Tips

### Efficient Navigation
- **Use Shift+Scroll for timeline**: Zoom time axis independently to see more frequency detail
- **Use Ctrl+Scroll for frequencies**: Zoom frequency axis to see harmonic structure
- **Combine zoom + pan**: Zoom in, then drag to explore - much faster than zoom out/in!

### Precision Work
- **Small scrolls**: Single scroll = 2.5x zoom (quick yet controllable)
- **Hold Alt**: Enable crosshair for exact time/frequency readings
- **Smooth panning**: Drag slowly for precise positioning

### Speed Navigation
- **Large scrolls**: 5 scrolls = 97.7x zoom (nearly 100x in 5 clicks!)
- **Quick drags**: Pan rapidly across spectrogram
- **Reset with 0**: Press 0 to return to full view anytime

---

## 🔧 Technical Details

### Zoom Factors
- **Base factor**: 2.5x per scroll (ultra sensitive!)
- **Compound**: Multiple scrolls multiply (3 scrolls = 2.5³ = 15.6x, 5 scrolls = 97.7x!)
- **Range**: 1% to 100% of data (100x zoom range)

### Pan Speed
- **Factor**: 1.0x (1:1 mouse to view movement)
- **Smooth**: Real-time updates at 60 FPS
- **Constrained**: Automatic boundary detection

### Axis Behavior
- **Independent**: Time and frequency can zoom separately
- **Linked**: Pan moves both axes together
- **Synchronized**: Axes update with view instantly

---

## 🎨 Visual Feedback

### While Panning
- Cursor: Normal (arrow)
- Crosshair: Disabled (prevents conflicts)
- View: Follows mouse smoothly

### While Hovering
- Cursor: Normal (arrow)
- Crosshair: Enabled with Alt key
- Readout: Time & frequency at cursor

### While Zooming
- Cursor: Normal (arrow)
- Zoom: Centers on mouse position
- Axes: Update tick labels automatically

---

## 🚀 Performance

All navigation operations run at **60 FPS** with no lag:
- Pan: Lightweight screen-to-world coordinate conversion
- Zoom: Simple camera rect scaling
- GPU rendering: VisPy handles all heavy lifting

---

## ❓ FAQ

**Q: Can I pan while zoomed out?**  
A: Yes! Pan works at any zoom level.

**Q: What happens if I drag too far?**  
A: Panning stops at data boundaries automatically.

**Q: Can I zoom on a specific point?**  
A: Yes! Zoom centers on your mouse cursor position.

**Q: How do I get back to full view quickly?**  
A: Press **0** to reset zoom instantly.

**Q: Can I zoom only time or only frequency?**  
A: Yes! Use **Shift+Scroll** (time) or **Ctrl+Scroll** (frequency).

**Q: Does panning affect performance?**  
A: No! Panning is extremely lightweight and runs at full 60 FPS.

---

## 🎉 Summary

The audio visualizer now provides **professional-grade navigation**:

✅ **Ultra-fast zoom** (2.5x per scroll - nearly 100x in 5 clicks!)  
✅ **Smooth pan** (click + drag)  
✅ **Independent axes** (Shift/Ctrl modifiers)  
✅ **Boundary constraints** (auto-limiting)  
✅ **High performance** (60 FPS always)  
✅ **Proper axis labels** (6pt fonts, aligned, no overlap)  

Navigate like a pro! 🎵📊

