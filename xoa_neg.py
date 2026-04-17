import os
from pathlib import Path
import random

neg_dir    = Path("data/negative")
neg_paths  = list(neg_dir.glob("*.jpg"))

# Xóa ngẫu nhiên 3664 ảnh
n_xoa      = len(neg_paths) - 16185   # 19849 - 16185 = 3664
xoa_list   = random.sample(neg_paths, n_xoa)

for p in xoa_list:
    os.remove(p)

# Kiểm tra lại
print(f"Negative còn lại: {len(list(neg_dir.glob('*.jpg')))}")
print(f"Positive        : {len(list(Path('data/positive').glob('*.jpg')))}")