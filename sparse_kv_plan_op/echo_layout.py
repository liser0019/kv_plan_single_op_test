"""Echo LRU 的 Host 布局计算；不依赖 torch/CANN。"""

I32_MAX = (1 << 31) - 1
CAPACITY_MAX = 4096
HIST_THREADS = SCAT_THREADS = 64
DEBUG_NPAD_MAX = 8192


def echo_partition(rows: int, available_cores: int):
    """返回 (实际核数, 每核行数, 尾核行数)，保证每核至少有一行。"""
    if rows <= 0 or available_cores <= 0:
        raise ValueError("rows and available_cores must be positive")
    budget = min(rows, available_cores)
    chunk = (rows + budget - 1) // budget
    blocks = (rows + chunk - 1) // chunk
    return blocks, chunk, rows - chunk * (blocks - 1)


def echo_workspace_elements(rows: int, topk: int, capacity: int, blocks: int):
    if not (rows > 0 and topk > 0 and 0 < capacity <= CAPACITY_MAX and 0 < blocks <= rows):
        raise ValueError("invalid Echo shape/block count")
    npad = 1 << capacity.bit_length()  # next_pow2(capacity + 1), including capacity=4096
    base = rows * (capacity + 1 + 4 + 3 * topk)
    elements = ((base + 1) & ~1) + blocks * (
        4 * npad + (HIST_THREADS + SCAT_THREADS) * 256 + 256 + DEBUG_NPAD_MAX)
    if elements > I32_MAX or rows * topk > I32_MAX:
        raise ValueError("workspace exceeds 32-bit kernel indexing")
    return elements
