import os
import glob

# 手动设置起止数字
l = 150
r = 199

# 目标文件夹路径
folder_path = "./data_output/cureus/"  # 替换为你的文件夹路径

# 获取所有 .csv 文件（排序以保持一致性）
csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

# 计算目标文件数量
expected_count = r - l + 1

# 校验文件数量是否匹配
if len(csv_files) < expected_count:
    raise ValueError(f"文件数量不足，需要 {expected_count} 个，但只找到 {len(csv_files)} 个。")

# 重命名文件
for i in range(expected_count):
    old_path = csv_files[i]
    new_filename = f"{l + i}.csv"
    new_path = os.path.join(folder_path, new_filename)
    os.rename(old_path, new_path)
    print(f"重命名: {os.path.basename(old_path)} -> {new_filename}")
