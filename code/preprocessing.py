import os
import re
import argparse
import chardet  # 导入chardet用于编码检测
from clean_gadget import clean_gadget


def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def normalize(path):
    setfolderlist = os.listdir(path)
    for setfolder in setfolderlist:
        catefolderlist = os.listdir(os.path.join(path, setfolder))
        for catefolder in catefolderlist:
            filepath = os.path.join(path, setfolder, catefolder)
            print(catefolder)
            pro_one_file(filepath)

def pro_one_file(filepath):
    # 检测文件编码
    with open(filepath, 'rb') as file:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']
        if encoding is None:  # 如果chardet无法检测到编码，默认使用utf-8
            encoding = 'utf-8'

    # 安全地以检测到的编码读取文件
    with open(filepath, 'r', encoding=encoding) as file:
        code = file.read()

    # 去除代码中的注释
    code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    with open(filepath, "w", encoding='utf-8') as file:  # 明确写文件时使用utf-8编码
        file.write(code.strip())

    # 读取并处理代码
    with open(filepath, 'r', encoding='utf-8') as file:  # 重新以utf-8编码打开文件
        org_code = file.readlines()
    # 假设clean_gadget是一个已定义的函数
    nor_code = clean_gadget(org_code)
    with open(filepath, "w", encoding='utf-8') as file:
        file.writelines(nor_code)

def main():
    args = parse_options()
    normalize(args.input)

if __name__ == '__main__':
    main()
