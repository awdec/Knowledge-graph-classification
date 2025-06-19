import os
import csv

# —— 1. 参数设置 —— #
l = 51
r = 100

input_dir = './data_output/cureus/2025'
output_dir = './data_output/cureus/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


# —— 2. 按照编号循环处理 —— #
for i in range(l, r + 1):
    input_file  = os.path.join(input_dir,  f'{i}_graph.csv')
    output_file = os.path.join(output_dir, f'{i}.csv')

    if not os.path.exists(input_file):
        print(f'[跳过] 文件不存在: {input_file}')
        continue

    # 用 csv.reader 直接将一行拆成若干列（默认逗号分隔）
    # 然后先把这些列全部拼接成一个“完整字符串”，再按 '|' 切分成新的列
    with open(input_file,  'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)   # 假设原 CSV 是逗号分隔
        writer = csv.writer(outfile)

        for row in reader:
            # —— 2.1 把这一行原始的所有列拼接成一个大字符串 —— #
            # 如果你想在列之间插入别的分隔符，比如空格，改成 ' '.join(row) 即可
            big_str = ''.join(row)
            # 示例：row = ['foo', 'bar', 'baz'] => big_str = 'foobarbaz'

            # —— 2.2 再用 '|' 切分刚刚拼接好的大字符串 —— #
            parts = big_str.split('|')
            # 如果 big_str = 'A|B|C' => parts = ['A','B','C']
            # 如果 big_str 中本身没有 '|'，则 parts = [big_str]

            # —— 2.3 (可选) 在这里对 parts 进行“后续操作” —— #
            # 例如：
            # for segment in parts:
            #     # 对 segment 做清洗、统计、NLP 分析……
            #     cleaned = segment.strip().lower()
            #     ……

            # —— 2.4 把切分后的结果写回到新的 CSV —— #
            writer.writerow(parts)

    print(f'[完成] 已处理: {input_file} -> {output_file}')
