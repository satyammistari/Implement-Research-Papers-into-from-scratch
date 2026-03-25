import time
import numpy as np
from typing import Callable

class RTX3050Profiler:
    def __init__(self, d: int, variant: str, N: int):
        self.d = d
        self.variant = variant
        self.N = N
        self.times = []
        self.operations = []

    def record(self, func: Callable, *args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        self.times.append(elapsed)
        return result

    def summary(self):
        if not self.times:
            return {}

        return {
            'variant': self.variant,
            'N': self.N,
            'd': self.d,
            'avg_time_ms': float(np.mean(self.times)),
            'std_time_ms': float(np.std(self.times)),
            'min_time_ms': float(np.min(self.times)),
            'max_time_ms': float(np.max(self.times)),
            'num_measurements': len(self.times)
        }

    def print_report(self):
        summary = self.summary()
        if not summary:
            print(f"[Profiler] No measurements for {self.variant} N={self.N}")
            return

        print(f"\n[Profiler] {self.variant} N={self.N} d={self.d}")
        print(f"  Avg time: {summary['avg_time_ms']:.2f}ms")
        print(f"  Std time: {summary['std_time_ms']:.2f}ms")
        print(f"  Min/Max:  {summary['min_time_ms']:.2f}/{summary['max_time_ms']:.2f}ms")
        print(f"  Measurements: {summary['num_measurements']}")
