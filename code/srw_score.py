import os
import mimetypes
import mutils
import copy

# 关键字到CWE的映射
keywords = {
    "strcpy": "CWE-121/CWE-122",  # 栈缓冲区溢出、堆缓冲区溢出
    "strncpy": "CWE-121/CWE-122",  # 同上，通常更安全，但仍需正确使用
    "memcpy": "CWE-121/CWE-122",  # 可能导致缓冲区溢出
    "memset": "CWE-121/CWE-122",  # 用于缓冲区操作，可能不当使用
    "sprintf": "CWE-134",  # 使用外部控制的格式字符串
    "printf": "CWE-134",  # 同上，如果格式字符串可控
    "system": "CWE-78",  # 操作系统命令注入
    "exec": "CWE-78",  # 同上
    "popen": "CWE-78",  # 同上
    "unsigned": "CWE-195",  # 符号到无符号转换错误
    "int": "CWE-195",  # 可能涉及符号到无符号转换
    "free": "CWE-590",  # 释放非堆内存
    "delete": "CWE-590",  # 同上
    "malloc": "CWE-122",  # 堆缓冲区溢出相关
    "calloc": "CWE-122",  # 堆缓冲区溢出相关
    "realloc": "CWE-122",  # 堆缓冲区溢出相关
    "access": "CWE-732",  # 权限控制问题
    "chmod": "CWE-732",  # 同上，改变文件权限
    "chown": "CWE-732",  # 同上，改变文件所有者
    "getenv": "CWE-78",  # 环境变量读取，可能影响命令注入
    "fopen": "CWE-676",  # 潜在的危险函数
    "read": "CWE-126",  # 缓冲区过读
    "write": "CWE-124",  # 缓冲区下写
    "bind": "CWE-755",  # 网络服务相关错误
    "listen": "CWE-755",  # 同上
    "recv": "CWE-754",  # 未检查的返回值
    "send": "CWE-754",  # 同上
    "socket": "CWE-755",  # 网络服务可能误用
    "connect": "CWE-755",  # 同上
}

def scan_file_for_vulnerabilities(file_path, keywords, default_cwe='other'):
    vulnerabilities = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            found_vulnerability = False
            for keyword, cwe in keywords.items():
                if keyword in line:
                    vulnerabilities.append((line_number, line.strip(), cwe))
                    found_vulnerability = True
                    break  # Stop checking other keywords once a match is found
            # If no keyword is found, mark it as other type of error
            if not found_vulnerability:
                vulnerabilities.append((line_number, line.strip(), default_cwe))

    return vulnerabilities


# 将文件类型判断后写入到txt
def getTypeScore():
    directory = './data/Vul/'
    filenames = os.listdir(directory)

    with open('typescore.txt', 'a') as wfile:
       for file in filenames:
            print(file)
            filePath = os.path.join(directory, file)
            vulnerabilities = scan_file_for_vulnerabilities(filePath, keywords)
            dic = {}
            for vulnerability in vulnerabilities:
                if (vulnerability[2] != 'other'):
                    dic[vulnerability[2]] = 1
                # print(vulnerability[2])
                # print(f"Line {vulnerability[0]}: '{vulnerability[1]}' Detected vulnerability type: {vulnerability[2]}")
            if len(dic) == 0:
                dic['other'] = 1
            else:
                print(dic)

            wfile.write(f"{file} ")
            for key, value in dic.items():
                wfile.write(f"{key} ")
            wfile.write('\n')
# 示例文件路径

# getTypeScore()

# 返回每个文件属于什么类型字典
def readTypeScore(directory):

    res_dic = {} #文件所属类型
    res_dic2 = {} #每个类型有哪些文件
    with open(directory, 'r') as rfile:
        for line in rfile:
            parts = line.strip().split(' ')
            key = parts[0]
            values = parts[1:]

            for value in values:
                if value in res_dic2:
                    res_dic2[value].append(key)
                else:
                    res_dic2[value] = []
            res_dic[key] = values
    return res_dic, res_dic2

# 每个c文件含有什么token
def calToken(dir1, dir2):
    token_ls = []
    dir_ls = [dir1, dir2]
    token_dic = {}
    all_dic = {}
    for mdir in dir_ls:
        token_dic = {}
        directory = mdir
        files = os.listdir(directory)

        a = []
        for file in files:
            print("calToken: " + str(file))
            file_path = os.path.join(directory, file)
            with open(file_path, 'r') as rfile:
                tmp_dic = {}
                for line in rfile:
                    tokens = mutils.tokenize_code_segment(line)
                    for token in tokens:
                        tmp_dic[token] = 1
                        all_dic[token] = 0
                token_dic[file] = tmp_dic
    token_ls.append(all_dic)
    token_ls.append(token_dic)

    return token_ls


def write_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 使用 split 函数只分割一次，并检查结果的长度是否为 2
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                key, value = parts
                # 对值进行进一步解析，如果值是字典，则解析为字典类型
                if value.startswith('{') and value.endswith('}'):
                    value = eval(value)  # 使用 eval 函数解析字符串为字典类型
                else:
                    value = int(value)  # 如果不是字典，则将值解析为整数
                # 将键值对添加到字典中
                result_dict[key] = value
            else:
                print(f"Ignoring invalid line: {line.strip()}")
    return result_dict

def typeDetailScore():
    dir1 = './data/Vul/'
    dir2 = './data/No-Vul/'
    token_ls = calToken(dir1, dir2)

    type_dic = {}
    for key, value in keywords.items():
        type_dic[value] = copy.deepcopy(token_ls[0])
    type_dic['other'] = copy.deepcopy(token_ls[0])

    directory = './typescore.txt'
    res_dic, res_dic2 = readTypeScore(directory)


    for key, value in type_dic.items():
        if key not in res_dic2:
            print(key)
            continue
        tmp_ls = res_dic2[key]
        for item in tmp_ls:
            if item not in token_ls[1]:
                continue
            for key2, value2 in value.items():
                t1 = token_ls[1][item]
                if(key2 in t1):
                    value[key2] += 1

    write_dict_to_txt(type_dic, './typescore_dic.txt')

# typeDetailScore()


# 返回各个类别中各个token得分
def handleDic():
    directory = './typescore.txt'
    type_num = {}
    res_dic, res_dic2 = readTypeScore(directory)


    # 计算每种类型错误文件数量总量
    for key, value in res_dic.items():
        for item in value:
            if item in type_num:
                type_num[item] += 1
            else:
                type_num[item] = 0

    score_dic = read_txt_to_dict('./typescore_dic.txt')

    for key, value in score_dic.items():

        for key2, value2 in value.items():
            if value2 != 0:
                value[key2] = type_num[key]/value[key2]

    return score_dic

def getMaxDic():
    typeDetailScore()
    # max_score = {}
    # dir1 = './data/Vul/'
    # dir2 = './data/No-Vul/'
    # token_ls = calToken(dir1, dir2)
    #
    # directory = './typescore.txt'
    # res_dic, res_dic2 = readTypeScore(directory)
    #
    # score_dic = handleDic()
    #
    # for key, value in token_ls[1].items():
    #     max_score[key] = copy.deepcopy(token_ls[1][key])
    #     for key2, value2 in max_score[key].items():
    #         for item in res_dic[key]:
    #             if (item in score_dic) and (key2 in score_dic[item]):
    #                 if value2 < score_dic[item][key2]:
    #                     max_score[key][key2] = score_dic[item][key2]
    # write_dict_to_txt(max_score, './typescore_dic_max.txt')

# getMaxDic()







