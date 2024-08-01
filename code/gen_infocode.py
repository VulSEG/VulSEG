import networkx as nx
import argparse
import os
import glob
from infercode.client.infercode_client import InferCodeClient

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    args = parser.parse_args()
    return args

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph

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

def image_generation(dot, front_name, mdic, infercode):

    try:
        pdg = graph_extraction(dot)
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()
        for label, all_code in labels_dict.items():
            code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            code = code.replace("static void", "void")
            labels_code[label] = code


        degree_channel = []
        closeness_channel = []
        katz_channel = []
        for label, code in labels_code.items():

            tmp_ls = []
            tmp_ls.append(code)
            line_vec2 = infercode.encode(tmp_ls)
            line_vec2=line_vec2[0]
            mdic[front_name][code] = line_vec2.tolist()


    except Exception as e:
        print("error:")
        print(e)
        return None

def write_to_pkl(dot,  existing_files,mdic,  infercode):

    dot_name = dot.split('/')[-1].split('.dot')[0]
    front_name = dot_name.split('\\')[-1]
    mdic[front_name] = {}
    if front_name in existing_files:
        return None

    else:
        print(front_name)
        image_generation(dot, front_name, mdic, infercode)




def main():
    args = parse_options()
    dir_name = args.input
    print("tf finish")
    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    infercode = InferCodeClient(language="c")
    infercode.init_from_config()

    existing_files = ['']


    dotfiles = glob.glob(dir_name + '*.dot')
    print(dotfiles)
    mdic = {}
    i = 0
    for dotfile in dotfiles:
        write_to_pkl(dotfile, existing_files, mdic,  infercode)

    write_dict_to_txt(mdic, "./into2test.txt")



if __name__ == '__main__':
    main()

