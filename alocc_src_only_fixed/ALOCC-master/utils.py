import time, torch, numpy, random
from functools import wraps

DEVICE = 'cuda' if torch.cuda.is_available() else ('xpu' if torch.xpu.is_available() else 'cpu')

_CURRENT_SEED = 42  # [A4-SEED]

def set_random_seed(seed=None):
    """[A4-SEED] If seed is None reuse the module-level _CURRENT_SEED (default 42);
    if an int is given, update _CURRENT_SEED then apply. Downstream no-arg calls
    (e.g. MNIST.py) thus inherit the user-chosen seed. Behavior with seed=None or
    set_random_seed() is bitwise identical to the historical fixed-42 implementation."""
    global _CURRENT_SEED
    if seed is not None:
        _CURRENT_SEED = int(seed)
    s = _CURRENT_SEED
    random.seed(s)
    numpy.random.seed(s)
    torch.manual_seed(s)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.xpu.is_available():
        torch.xpu.manual_seed_all(s)

set_random_seed()

def timer(func):
    """简单的计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper