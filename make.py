import os
import chardet


# 安装 chardet：pip install chardet
def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result['encoding']


def convert_csv_to_utf8(folder_path, overwrite=True):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            full_path = os.path.join(folder_path, filename)
            try:
                original_encoding = detect_encoding(full_path)
                print(f"[INFO] 正在处理：{filename}，检测编码：{original_encoding}")

                # 读取原文件内容
                with open(full_path, 'r', encoding=original_encoding, errors='ignore') as f:
                    content = f.read()

                # 保存为 UTF-8
                if overwrite:
                    save_path = full_path
                else:
                    save_path = os.path.join(folder_path, f"utf8_{filename}")

                with open(save_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(content)

                print(f"[成功] 转换为 UTF-8：{save_path}")
            except Exception as e:
                print(f"[失败] 处理文件出错：{filename}，错误信息：{e}")


# 用法
folder = "./data_output/cureus"
convert_csv_to_utf8(folder, overwrite=True)  # 设置为 False 会保留原文件，另存为 utf8_开头的副本
