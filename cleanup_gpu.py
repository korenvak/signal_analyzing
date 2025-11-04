"""
GPU Memory Cleanup Utility
Run this to clean GPU memory before starting the application
"""

import sys
import gc

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not available - no GPU cleanup needed")
    sys.exit(0)

def cleanup_gpu_memory():
    """Thoroughly clean GPU memory."""
    print("=" * 70)
    print("GPU MEMORY CLEANUP UTILITY")
    print("=" * 70)
    
    # Get device info
    device = cp.cuda.Device()
    device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8')
    print(f"\nGPU Device: {device_name}")
    
    # Before cleanup
    mem_before = device.mem_info
    used_before = (mem_before[1] - mem_before[0]) / (1024**2)
    total = mem_before[1] / (1024**2)
    
    print(f"\n[DATA] Memory BEFORE cleanup:")
    print(f"   Used: {used_before:.1f} MB / {total:.1f} MB")
    print(f"   Free: {mem_before[0] / (1024**2):.1f} MB")
    print(f"   Utilization: {(used_before / total * 100):.1f}%")
    
    # Perform cleanup
    print(f"\n[CLEANUP] Cleaning up...")
    
    # 1. Synchronize all streams
    cp.cuda.Stream.null.synchronize()
    print("   [OK] Synchronized GPU streams")
    
    # 2. Python garbage collection
    collected = gc.collect()
    print(f"   [OK] Python GC collected {collected} objects")
    
    # 3. Free CuPy memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    mempool_before = mempool.used_bytes() / (1024**2)
    
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    print(f"   [OK] Freed {mempool_before:.1f} MB from memory pools")
    
    # 4. Another GC pass
    gc.collect()
    
    # After cleanup
    mem_after = device.mem_info
    used_after = (mem_after[1] - mem_after[0]) / (1024**2)
    freed = used_before - used_after
    
    print(f"\n[DATA] Memory AFTER cleanup:")
    print(f"   Used: {used_after:.1f} MB / {total:.1f} MB")
    print(f"   Free: {mem_after[0] / (1024**2):.1f} MB")
    print(f"   Utilization: {(used_after / total * 100):.1f}%")
    
    print(f"\n[OK] Cleanup complete!")
    print(f"   Freed: {freed:.1f} MB")
    print(f"   Now available: {mem_after[0] / (1024**2):.1f} MB")
    
    if freed > 0:
        print(f"\n[SUCCESS] Successfully freed {freed:.1f} MB of GPU memory")
    else:
        print(f"\n[INFO] GPU memory was already clean")
    
    print("\n" + "=" * 70)
    return freed

if __name__ == '__main__':
    try:
        cleanup_gpu_memory()
    except Exception as e:
        print(f"\n[ERROR] Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

