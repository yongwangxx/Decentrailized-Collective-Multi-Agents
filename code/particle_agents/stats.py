"""Small statistics helpers."""
import math


def wilson(k, n, z=1.96):
    """Wilson score interval (lower, upper) for k successes in n trials. z = 1.96 gives 95%, z = 1 gives 68%."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def frac(k, n):
    """'k/n (xx.x%)' for tables."""
    return f"{k}/{n}" + (f" ({100 * k / n:.1f}%)" if n else "")
