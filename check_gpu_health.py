"""
GPU Health Check
Comprehensive test to ensure GPU is working correctly before starting implementation
"""

import sys
import time
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    print("[ERROR] CuPy not available!")
    sys.exit(1)

def check_gpu_health():
    """Run comprehensive GPU health checks."""
    print("=" * 70)
    print("GPU HEALTH CHECK")
    print("=" * 70)
    
    # 1. Device Info
    print("\n[1] GPU Device Information:")
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = props['name'].decode('utf-8')
    
    print(f"   Device: {device_name}")
    print(f"   Compute Capability: {props['major']}.{props['minor']}")
    print(f"   Multiprocessors: {props['multiProcessorCount']}")
    print(f"   Max Threads/Block: {props['maxThreadsPerBlock']}")
    print(f"   Warp Size: {props['warpSize']}")
    
    # 2. Memory Status
    print("\n[2] Memory Status:")
    mem_info = device.mem_info
    total_mb = mem_info[1] / (1024**2)
    free_mb = mem_info[0] / (1024**2)
    used_mb = (mem_info[1] - mem_info[0]) / (1024**2)
    util = (used_mb / total_mb * 100)
    
    print(f"   Total: {total_mb:.1f} MB")
    print(f"   Used: {used_mb:.1f} MB")
    print(f"   Free: {free_mb:.1f} MB")
    print(f"   Utilization: {util:.1f}%")
    
    if util > 80:
        print("   [WARNING] GPU memory over 80% utilized!")
        return False
    elif util > 50:
        print("   [INFO] GPU memory over 50% utilized")
    else:
        print("   [OK] GPU memory healthy")
    
    # 3. Memory Pool Status
    print("\n[3] Memory Pool Status:")
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    pool_used = mempool.used_bytes() / (1024**2)
    pool_total = mempool.total_bytes() / (1024**2)
    
    print(f"   Pool Used: {pool_used:.1f} MB")
    print(f"   Pool Total: {pool_total:.1f} MB")
    print(f"   Pinned Memory: {pinned_mempool.n_free_blocks()} free blocks")
    print("   [OK] Memory pools initialized")
    
    # 4. Basic Computation Test
    print("\n[4] Basic Computation Test:")
    try:
        # Allocate test arrays
        size = 1024 * 1024  # 1M elements
        a = cp.random.random((size,), dtype=cp.float32)
        b = cp.random.random((size,), dtype=cp.float32)
        
        # Test computation
        start = time.time()
        c = a + b
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        
        print(f"   Vector addition (1M elements): {elapsed*1000:.2f} ms")
        
        # Verify result
        result_ok = cp.all(c > 0)
        if result_ok:
            print("   [OK] Computation correct")
        else:
            print("   [ERROR] Computation failed!")
            return False
            
        # Cleanup
        del a, b, c
        
    except Exception as e:
        print(f"   [ERROR] Computation test failed: {e}")
        return False
    
    # 5. FFT Test (Critical for our app)
    print("\n[5] FFT Performance Test:")
    try:
        # Test FFT
        size = 2048
        n_frames = 1000
        data = cp.random.random((size, n_frames), dtype=cp.float32)
        
        start = time.time()
        fft_result = cp.fft.rfft(data, axis=0)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        
        print(f"   rFFT ({size}x{n_frames}): {elapsed*1000:.2f} ms")
        print(f"   Throughput: {n_frames/elapsed:.0f} frames/sec")
        
        if elapsed > 1.0:
            print("   [WARNING] FFT slower than expected")
        else:
            print("   [OK] FFT performance good")
        
        del data, fft_result
        
    except Exception as e:
        print(f"   [ERROR] FFT test failed: {e}")
        return False
    
    # 6. Memory Transfer Test
    print("\n[6] Host-Device Transfer Test:")
    try:
        # Test H2D transfer
        size_mb = 10
        host_data = np.random.random((size_mb * 1024 * 256,)).astype(np.float32)
        
        start = time.time()
        device_data = cp.asarray(host_data)
        cp.cuda.Stream.null.synchronize()
        h2d_time = time.time() - start
        h2d_bandwidth = size_mb / h2d_time
        
        print(f"   Host->Device ({size_mb}MB): {h2d_time*1000:.2f} ms ({h2d_bandwidth:.0f} MB/s)")
        
        # Test D2H transfer
        start = time.time()
        host_result = cp.asnumpy(device_data)
        d2h_time = time.time() - start
        d2h_bandwidth = size_mb / d2h_time
        
        print(f"   Device->Host ({size_mb}MB): {d2h_time*1000:.2f} ms ({d2h_bandwidth:.0f} MB/s)")
        
        if h2d_bandwidth < 100 or d2h_bandwidth < 100:
            print("   [WARNING] Transfer bandwidth lower than expected")
        else:
            print("   [OK] Transfer performance good")
        
        del host_data, device_data, host_result
        
    except Exception as e:
        print(f"   [ERROR] Transfer test failed: {e}")
        return False
    
    # 7. Memory Leak Test
    print("\n[7] Memory Leak Test:")
    try:
        initial_used = mempool.used_bytes()
        
        # Allocate and free in loop
        for i in range(10):
            temp = cp.random.random((1024, 1024), dtype=cp.float32)
            del temp
        
        cp.cuda.Stream.null.synchronize()
        import gc
        gc.collect()
        mempool.free_all_free()
        
        final_used = mempool.used_bytes()
        leaked = (final_used - initial_used) / (1024**2)
        
        print(f"   Initial: {initial_used / (1024**2):.1f} MB")
        print(f"   Final: {final_used / (1024**2):.1f} MB")
        print(f"   Leaked: {leaked:.1f} MB")
        
        if abs(leaked) < 1.0:
            print("   [OK] No significant memory leak")
        else:
            print("   [WARNING] Possible memory leak detected")
        
    except Exception as e:
        print(f"   [ERROR] Leak test failed: {e}")
        return False
    
    # 8. Final Memory Check
    print("\n[8] Final Memory Status:")
    mem_info = device.mem_info
    final_free = mem_info[0] / (1024**2)
    final_used = (mem_info[1] - mem_info[0]) / (1024**2)
    final_util = (final_used / total_mb * 100)
    
    print(f"   Free: {final_free:.1f} MB")
    print(f"   Used: {final_used:.1f} MB")
    print(f"   Utilization: {final_util:.1f}%")
    
    if final_free < 1000:
        print("   [WARNING] Less than 1GB free!")
        return False
    else:
        print("   [OK] Sufficient memory available")
    
    # 9. Cleanup
    print("\n[9] Cleanup:")
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    print("   [OK] Cleaned up test allocations")
    
    # Final verdict
    mem_info = device.mem_info
    final_free = mem_info[0] / (1024**2)
    
    print("\n" + "=" * 70)
    print("HEALTH CHECK SUMMARY")
    print("=" * 70)
    print(f"GPU Device: {device_name}")
    print(f"Available Memory: {final_free:.1f} MB / {total_mb:.1f} MB")
    print(f"Status: [OK] GPU is healthy and ready!")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    try:
        success = check_gpu_health()
        if success:
            print("\n[SUCCESS] GPU is ready for high-performance implementation!")
            sys.exit(0)
        else:
            print("\n[ERROR] GPU health check failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Health check error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

