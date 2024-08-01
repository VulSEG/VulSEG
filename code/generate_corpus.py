import os
import re

def process_code(code):
    # 移除大括号
    code = code.replace('{', '').replace('}', '')

    # 添加空格到括号、方括号等特殊符号前后
    code = re.sub(r'([()\[\],;])', r' \1 ', code)

    # 替换多余的空格（多个空格替换为一个）
    code = re.sub(r'\s+', ' ', code)

    # 处理特殊情况，如多个标点符号间的空格
    code = re.sub(r'\s*([()\[\],;])\s*', r' \1 ', code)

    # 去除可能出现的行首行尾多余空格
    code = re.sub(r'^\s+|\s+$', '', code)

    return code

def process_files(source_directory, destination_directory):
    # 确保输出目录存在
    os.makedirs(destination_directory, exist_ok=True)

    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            print(file)
            if file.endswith('.c') or file.endswith('.cpp'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # 处理代码
                processed_code = process_code(code)

                # 构建新文件路径
                new_file_path = os.path.join(destination_directory, os.path.splitext(file)[0] + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_code)

# 调用函数
source_directories = ['./data/Vul', './data/No-Vul']
destination_directory = './corpus'

for source_directory in source_directories:
    process_files(source_directory, destination_directory)