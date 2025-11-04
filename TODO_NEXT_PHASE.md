# Next Phase - Optimization & GUI Improvements

## A. Performance Optimizations Remaining

### 1. Full Tile System Integration
**Current**: Using direct computation (works but downsamples to 16K pixels)  
**Needed**: Integrate TextureAtlas to render tiles without OpenGL limits  
**Benefit**: True unlimited file size, no downsampling ever  
**Effort**: 1-2 days

### 2. GPU Stream Management
**Current**: Single stream, sequential operations  
**Needed**: 3 CUDA streams (transfer/compute/readback) with double-buffering  
**Benefit**: 2-3x throughput improvement, hide transfer latency  
**Effort**: 1 day

### 3. Shader-Based Colormap
**Current**: CPU normalization and colormap  
**Needed**: Move colormap/dB scaling to fragment shader  
**Benefit**: Instant parameter changes, zero CPU/GPU transfer  
**Effort**: 4-6 hours

### 4. F-K Transform Optimization
**Current**: Disabled (causes 96GB allocation)  
**Needed**: Precomputed phase tables, online statistics, no 3D storage  
**Benefit**: Working F-K transform with reasonable memory  
**Effort**: 1 day

### 5. Memory Pool Reuse
**Current**: Workspace pools created but not fully integrated  
**Needed**: Use workspace pools throughout all engines  
**Benefit**: Zero malloc/free overhead, predictable memory  
**Effort**: 4-6 hours

---

## B. GUI Issues to Fix

### 1. Axis Labels Not Visible
**Issue**: Text visuals created but don't appear on screen  
**Cause**: Possible z-order issue, coordinate system mismatch, or VisPy Text rendering  
**Fix Needed**: Debug VisPy Text positioning, try different coordinate systems  
**Effort**: 2-4 hours

### 2. Separate Axis Zoom Not Working
**Issue**: Shift/Ctrl + Mouse Wheel doesn't zoom individual axes  
**Cause**: Event modifiers not detected or event.handled not preventing default  
**Fix Needed**: Debug VisPy event system, try alternative key bindings  
**Effort**: 2-4 hours

### 3. Camera Constraints Not Working
**Issue**: Can still pan/zoom to empty black regions  
**Cause**: Camera.set_range() not enforcing bounds, or timing issue with updates  
**Fix Needed**: Investigate VisPy PanZoomCamera internals, implement custom camera  
**Effort**: 4-6 hours

### 4. No Grid Lines or Tick Marks
**Issue**: No visual reference for scale  
**Needed**: Grid lines, tick marks with values  
**Benefit**: Professional appearance, easier to read values  
**Effort**: 4-6 hours

---

## C. Features to Add

### 1. Colormap Selection
**Status**: Dropdown exists but not connected  
**Needed**: Hook up colormap changes to VisPy Image.cmap  
**Effort**: 1 hour

### 2. dB Range Controls
**Status**: Sliders exist but not connected  
**Needed**: Apply dB range to normalization  
**Effort**: 1 hour

### 3. Export Functionality
**Needed**: Export current view as image, export data as NPY/CSV  
**Benefit**: Save results for reports/papers  
**Effort**: 2-3 hours

### 4. Measurement Tools
**Needed**: Crosshair with readout, time/frequency cursors, peak detection  
**Benefit**: Quantitative analysis  
**Effort**: 4-6 hours

### 5. Real-Time Streaming
**Needed**: Process audio file in chunks while displaying  
**Benefit**: Start viewing while still loading large files  
**Effort**: 2-3 days

---

## D. Code Cleanup Needed

### Files to Delete:
- All debug/test PNG files
- Temporary test scripts
- Old MD documentation files (keep only one)
- Test WAV files
- Unused debug scripts

### Files to Keep:
- Core application code
- Utilities (cleanup_gpu.py, check_gpu_health.py)
- Main documentation (ORIENTATION_AND_PERFORMANCE_FIX.md)
- Test suite (test_phase2_integration.py)

---

## Priority Ranking

### CRITICAL (Must Have):
1. ✅ Orientation fix - DONE
2. ✅ Performance optimization - DONE (batched FFT)
3. ✅ Memory management - DONE
4. ⏳ Full tile integration - HIGH PRIORITY
5. ⏳ GUI fixes (labels, zoom, constraints) - HIGH PRIORITY

### IMPORTANT (Should Have):
1. Grid lines and tick marks
2. Colormap/dB controls hookup
3. Export functionality
4. F-K transform fix

### NICE TO HAVE (Could Have):
1. GPU streams
2. Shader-based rendering
3. Real-time streaming
4. Measurement tools

---

## Estimated Time to Complete

### Minimum Viable (GUI fixes only):
- Fix axis labels: 2-4 hours
- Fix zoom controls: 2-4 hours
- Fix camera constraints: 4-6 hours
**Total**: 8-14 hours

### Full Feature Set:
- GUI fixes: 8-14 hours
- Tile integration: 16 hours
- Additional features: 24 hours
**Total**: 48-54 hours (1-2 weeks)

---

## Recommended Next Steps

1. **Clean up project** (remove temp files)
2. **Push to Git** (preserve current working state)
3. **Choose focus**:
   - Option A: Fix GUI issues first (labels, zoom, constraints)
   - Option B: Complete tile integration first (unlimited resolution)
   - Option C: Use as-is and move to other features

**Current state is functional** - main orientation issue fully resolved!

