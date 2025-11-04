import threading
import queue
import uuid
import time
import asyncio
from typing import Callable, Any, Dict, Optional, List
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import multiprocessing as mp

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class Task:
    """Represents a computation task with metadata and progress tracking."""
    
    def __init__(self, func: Callable, args: tuple = (), kwargs: Dict = None,
                 priority: TaskPriority = TaskPriority.NORMAL,
                 progress_callback: Optional[Callable] = None,
                 completion_callback: Optional[Callable] = None,
                 description: str = ""):
        
        self.id = str(uuid.uuid4())
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.description = description
        
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.future = None
        
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value > other.priority.value
    
    def update_progress(self, progress: float, message: str = ""):
        """Update task progress (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
        if self.progress_callback:
            try:
                self.progress_callback(self.id, self.progress, message)
            except Exception as e:
                print(f"Progress callback error: {e}")
    
    def complete(self, result: Any = None):
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.end_time = time.time()
        
        if self.completion_callback:
            try:
                self.completion_callback(self.id, result, None)
            except Exception as e:
                print(f"Completion callback error: {e}")
    
    def fail(self, error: Exception):
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.end_time = time.time()
        
        if self.completion_callback:
            try:
                self.completion_callback(self.id, None, error)
            except Exception as e:
                print(f"Completion callback error: {e}")

class TaskManager:
    """High-performance task manager for background computation."""
    
    def __init__(self, max_threads: int = None, max_processes: int = None):
        self.max_threads = max_threads or min(8, mp.cpu_count())
        self.max_processes = max_processes or max(1, mp.cpu_count() - 1)
        
        self._thread_executor = ThreadPoolExecutor(max_workers=self.max_threads)
        self._process_executor = ProcessPoolExecutor(max_workers=self.max_processes)
        
        self._tasks = {}
        self._task_queue = queue.PriorityQueue()
        self._active_tasks = {}
        self._lock = threading.RLock()
        
        self._shutdown = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
    def submit_task(self, func: Callable, args: tuple = (), kwargs: Dict = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   use_process: bool = False,
                   progress_callback: Optional[Callable] = None,
                   completion_callback: Optional[Callable] = None,
                   description: str = "") -> str:
        """Submit a task for background execution."""
        
        task = Task(func, args, kwargs, priority, 
                   progress_callback, completion_callback, description)
        
        with self._lock:
            self._tasks[task.id] = task
            self._task_queue.put((priority.value, time.time(), task.id, use_process))
            
        return task.id
    
    def submit_gpu_task(self, func: Callable, args: tuple = (), kwargs: Dict = None,
                       priority: TaskPriority = TaskPriority.HIGH,
                       progress_callback: Optional[Callable] = None,
                       completion_callback: Optional[Callable] = None,
                       description: str = "") -> str:
        """Submit a GPU-accelerated task with high priority."""
        
        return self.submit_task(func, args, kwargs, priority, False,
                              progress_callback, completion_callback, 
                              f"GPU: {description}")
    
    def _worker_loop(self):
        """Main worker loop for task processing."""
        while not self._shutdown:
            try:
                priority, submit_time, task_id, use_process = self._task_queue.get(timeout=1.0)
                
                with self._lock:
                    if task_id not in self._tasks:
                        continue
                    
                    task = self._tasks[task_id]
                    if task.status != TaskStatus.PENDING:
                        continue
                    
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()
                    
                    executor = self._process_executor if use_process else self._thread_executor
                    
                    future = executor.submit(self._execute_task, task_id)
                    task.future = future
                    self._active_tasks[task_id] = task
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker loop error: {e}")
    
    def _execute_task(self, task_id: str):
        """Execute a single task with error handling."""
        try:
            with self._lock:
                task = self._tasks.get(task_id)
                if not task or task.status != TaskStatus.RUNNING:
                    return
            
            # Create progress updater
            def progress_updater(progress: float, message: str = ""):
                task.update_progress(progress, message)
            
            # Add progress updater to kwargs if the function accepts it
            kwargs = task.kwargs.copy()
            import inspect
            sig = inspect.signature(task.func)
            if 'progress_callback' in sig.parameters:
                kwargs['progress_callback'] = progress_updater
            
            result = task.func(*task.args, **kwargs)
            task.complete(result)
            
        except Exception as e:
            task.fail(e)
        
        finally:
            with self._lock:
                if task_id in self._active_tasks:
                    del self._active_tasks[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
            
            elif task.status == TaskStatus.RUNNING and task.future:
                success = task.future.cancel()
                if success:
                    task.status = TaskStatus.CANCELLED
                    if task_id in self._active_tasks:
                        del self._active_tasks[task_id]
                return success
            
            return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.status if task else None
    
    def get_task_progress(self, task_id: str) -> float:
        """Get current progress of a task (0.0 to 1.0)."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.progress if task else 0.0
    
    def get_task_result(self, task_id: str) -> Any:
        """Get result of completed task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
            elif task and task.status == TaskStatus.FAILED:
                raise task.error
            return None
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for task completion."""
        start_time = time.time()
        while True:
            status = self.get_task_status(task_id)
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
    
    def cancel_view_tasks(self, view_type: str):
        """Cancel all tasks related to a specific view type."""
        cancelled_count = 0
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if view_type in task.description:
                    if self.cancel_task(task_id):
                        cancelled_count += 1
        return cancelled_count
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of currently active tasks."""
        with self._lock:
            return [
                {
                    'id': task.id,
                    'description': task.description,
                    'status': task.status.value,
                    'progress': task.progress,
                    'start_time': task.start_time,
                    'priority': task.priority.value
                }
                for task in self._active_tasks.values()
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        with self._lock:
            status_counts = {}
            for task in self._tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_tasks': len(self._tasks),
                'active_tasks': len(self._active_tasks),
                'queued_tasks': self._task_queue.qsize(),
                'thread_pool_size': self.max_threads,
                'process_pool_size': self.max_processes,
                'status_breakdown': status_counts
            }
    
    def cleanup_completed_tasks(self, keep_recent: int = 100):
        """Remove old completed tasks to free memory."""
        with self._lock:
            completed_tasks = [
                (task.end_time or 0, task_id) 
                for task_id, task in self._tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            
            completed_tasks.sort(reverse=True)
            
            if len(completed_tasks) > keep_recent:
                to_remove = completed_tasks[keep_recent:]
                for _, task_id in to_remove:
                    del self._tasks[task_id]
    
    def shutdown(self, wait: bool = True):
        """Shutdown the task manager."""
        self._shutdown = True
        
        if wait:
            for task_id in list(self._active_tasks.keys()):
                self.cancel_task(task_id)
        
        self._thread_executor.shutdown(wait=wait)
        self._process_executor.shutdown(wait=wait)
        
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)