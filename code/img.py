
import html
import re
import time

import networkx as nx
import numpy as np
import argparse
import os
import torch
import pickle
import glob
from multiprocessing import Pool
from functools import partial
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaModel
import mutils
import json
from sklearn.preprocessing import StandardScaler
import traceback



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

def load_word2vec_model(model_path):
    """加载 Word2Vec 模型"""
    model = Word2Vec.load(model_path)
    return model

def calculate_tfidf_for_corpus(directory_path):
    """计算目录中所有文本文件的TF-IDF分数"""
    # 使用正则表达式来定义分词器，匹配所有非空白字符序列
    token_pattern = r'\S+'

    # 初始化TF-IDF向量化器，使用自定义的token_pattern
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=token_pattern)

    # 存储文本内容的列表
    corpus = []
    # 存储文件名的列表
    filenames = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # 确保处理的是文本文件
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                corpus.append(text)
                filenames.append(filename)

    # 计算整个语料库的TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 初始化存储TF-IDF分数的字典
    tfidf_scores_dict = {}

    # 提取每个文档的TF-IDF分数，并以文件名为键存储在字典中
    feature_names = vectorizer.get_feature_names_out()
    for doc_idx, filename in enumerate(filenames):
        # 提取每个文档的TF-IDF向量并转换为字典，键为单词，值为TF-IDF分数
        doc_vector = tfidf_matrix[doc_idx].todense().A1
        word_scores = dict(zip(feature_names, doc_vector))
        # 过滤掉TF-IDF分数为0的项
        word_scores = {word: score for word, score in word_scores.items() if score > 0}
        front_name = filename.split('.')[0]
        tfidf_scores_dict[front_name] = word_scores

    return tfidf_scores_dict

def sentence_embedding(sentence,  model, tfidf_scores):
    words = mutils.tokenize_code_segment(sentence)
    embedding = np.zeros((model.vector_size,))  # 初始化嵌入向量
    weight_sum = 0  # 初始化权重和
    all_scores = 0.0

    for word in words:
        if word in model.wv and word in tfidf_scores:
            embedding += model.wv[word] * tfidf_scores[word]
            weight_sum += 1
        if word not in tfidf_scores:
           continue


    if weight_sum > 0:
        embedding /= weight_sum  # 使用加权和的平均值作为句子嵌入

    return embedding


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

# def sentence_embedding(sentence):
#     emb = sent2vec_model.embed_sentence(sentence)
#     return emb[0]



# #  1-2 2-3 1-3
# def create_crg(labels_dict, window_size=3):
#
#     """根据代码标签创建上下文关系图（CRG），使用指定的滑动窗口大小"""
#     crg = nx.Graph()  # 使用无向图，如果需要双向连接
#     labels = sorted(labels_dict.keys())  # 假设标签已经按逻辑顺序排序
#
#     for i, label in enumerate(labels):
#         crg.add_node(label, code=labels_dict[label])
#         # 前后窗口大小为window_size的边，不包括当前节点
#         start = max(0, i - window_size)  # 确定窗口的起始位置
#         end = min(len(labels), i + window_size + 1)  # 确定窗口的结束位置
#         for j in range(start, end):
#             if i != j:
#                 crg.add_edge(label, labels[j])  # 增加从当前节点到窗口中其他节点的边
#
#     return crg


# 1-3
def create_crg(labels_dict, window_size=3):
    """根据代码标签创建上下文关系图（CRG），只连接每个窗口的第一个和最后一个节点"""
    crg = nx.Graph()  # 使用无向图，如果需要双向连接
    labels = sorted(labels_dict.keys())  # 假设标签已经按逻辑顺序排序

    for i, label in enumerate(labels):
        crg.add_node(label, code=labels_dict[label])
        # 确定窗口的起始和结束位置
        start = max(0, i - window_size)
        end = min(len(labels), i + window_size + 1)

        # 检查是否存在至少两个节点以形成边
        if end - start > 1:
            # 只连接窗口的第一个和最后一个节点
            crg.add_edge(labels[start], labels[end - 1])

    return crg



def merge_crg_cpg(pdg, crg):
    for node, data in crg.nodes(data=True):
        if node not in pdg:
            pdg.add_node(node, **data)
    for src, dst in crg.edges():
        if not pdg.has_edge(src, dst):
            pdg.add_edge(src, dst)
    return pdg
def image_generation(dot, word2vec_model,  tfidf_score_dict, front_name, mdic1, mdic2, flag2):
    try:
        pdg = graph_extraction(dot)
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()
        if flag2 == '1':
            for label, all_code in labels_dict.items():
                code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
                code = code.replace("static void", "void")
                code = re.sub(r'\)<SUB>\d+</SUB', '', code)
                labels_code[label] = code

        else:
            for label, all_code in labels_dict.items():
                # 使用正则表达式提取“VAR1 = -1”这样的赋值部分
                code_match = re.search(r',([^,]+?)<SUB>', all_code)
                if code_match:
                    code = code_match.group(1).strip()  # 去除前后可能的空格
                    labels_code[label] = code
                else:
                    print(f"Could not extract assignment from: {all_code}")
                    labels_code[label] = "default or error handling code"


        crg = create_crg(labels_code)  # 创建CRG
        merged_graph = merge_crg_cpg(pdg, crg)  # 合并CRG和CPG
        # merged_graph = pdg
        # 计算网络中心性

        #print(labels_code)
        degree_cen_dict = nx.degree_centrality(merged_graph)
        closeness_cen_dict = nx.closeness_centrality(merged_graph)
        G = nx.DiGraph()
        G.add_nodes_from(merged_graph.nodes())
        G.add_edges_from(merged_graph.edges())
        katz_cen_dict = nx.katz_centrality(G)

        degree_channel = []
        closeness_channel = []
        katz_channel = []

        degree_channel2 = []
        closeness_channel2 = []
        katz_channel2 = []
        i = 0
        for label, code in labels_code.items():
            i = i + 1

            line_vec = sentence_embedding(code, word2vec_model, tfidf_score_dict)  # 使用 Word2Vec 模型编码
            line_vec = np.array(line_vec)

            # 初始化第二个向量
            line_vec2 = np.zeros(100)  # 创建一个与line_vec形状相同，初始化为0的向量
            if flag2 == "1":
                # 检查并获取第二个向量
                if front_name in mdic1 and code in mdic1[front_name]:
                    line_vec2 = np.array(mdic1[front_name][code])

                elif front_name in mdic2 and code in mdic2[front_name]:
                    line_vec2 = np.array(mdic2[front_name][code])

                else:
                    print(i)
                    print(front_name)
                # print(code)
                # if front_name in mdic1:
                #     print("Keys in mdic1:", mdic1[front_name].keys())
                # else:
                #     print("Keys in mdic2:", mdic2[front_name].keys())
                # print("not in infocode dic")




            degree_cen = degree_cen_dict[label]
            degree_channel.append(degree_cen * line_vec)

            closeness_cen = closeness_cen_dict[label]
            closeness_channel.append(closeness_cen * line_vec)

            katz_cen = katz_cen_dict[label]
            katz_channel.append(katz_cen * line_vec)


            degree_cen2 = degree_cen_dict[label]
            degree_channel2.append(degree_cen * line_vec2)

            closeness_cen2 = closeness_cen_dict[label]
            closeness_channel2.append(closeness_cen * line_vec2)

            katz_cen2 = katz_cen_dict[label]
            katz_channel2.append(katz_cen * line_vec2)


        return (degree_channel, closeness_channel, katz_channel, degree_channel2, closeness_channel2, katz_channel2)
    except Exception as e:
        print("error:")
        print(e)
        traceback.print_exc()
        return None

def write_to_pkl(dot, out, existing_files, word2vec_model, tfidf_scores_dict, mdic1, mdic2, flag2):
    print("write_to_pkl111")
    dot_name = dot.split('/')[-1].split('.dot')[0]
    front_name = dot_name.split('\\')[-1]
    if front_name in existing_files:
        return None
    elif front_name not in tfidf_scores_dict:
        return None
    else:
        if flag2 == "1":
            print("pdg")
        else:
            print("cfg")
        print(front_name)
        channels = image_generation(dot, word2vec_model, tfidf_scores_dict[front_name], front_name, mdic1, mdic2, flag2)
        if channels == None:
            print("channels None")
            return None
        else:
            (degree_channel, closeness_channel, katz_channel, degree_channel2, closeness_channel2, katz_channel2) = channels
            out_pkl = out + dot_name + '.pkl'
            data = [degree_channel, closeness_channel, katz_channel, degree_channel2, closeness_channel2, katz_channel2]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)
            print("write")

def finish_file(file_path):
    filenames = os.listdir(file_path)
    with open('./data/outputs/file.txt', 'w') as file:
        for filename in filenames:
            front_name = filename.split('.')[0]
            file.write(str(front_name) +'\n')
    file.close()



def write_dic():
    directory_path = ".\\data\\corpus"
    tfidf_scores_dict = calculate_tfidf_for_corpus(directory_path)
    with open('./data/dic.txt', 'w') as file:
        import json
        json.dump(tfidf_scores_dict, file)

def main():


    mdic1 = read_txt_to_dict("./into2test.txt")
    mdic2 = read_txt_to_dict("./into2test2.txt")
    trained_model_path = './model/word2vec2_model3.model'
    # 加载 Word2Vec 模型
    global word2vec_model
    word2vec_model = load_word2vec_model(trained_model_path)
    tfidf_scores_dict = {}
    with open('./dic2.txt', 'r') as file:
        tfidf_scores_dict = json.load(file)
    flags = ['1', '2']
    flag2s = ['1', '2']
    for flag2 in flag2s:
        for flag in flags:
            sub_name = "Vul"
            if flag != '1':
                sub_name = "No-Vul"
            dir_name = './data/pdgs/' + sub_name + '/'
            dir_name2 = './data/cfgs/' + sub_name + '/'
            if flag2 == '2':
                out_path = './data/outputs/all_pdg2/' + sub_name + '/'
            else:
                out_path = './data/outputs/all_cfg2/' + sub_name + '/'

            # print(tfidf_scores_dict)

            if flag2 == '1':
                dotfiles = glob.glob(dir_name + '*.dot')
            else:
                dotfiles = glob.glob(dir_name2 + '*.dot')


            if not os.path.exists(out_path):
                os.makedirs(out_path)

            existing_files = ['']

            # print(dotfiles)
            pool = Pool(10)
            pool.map(partial(write_to_pkl, out=out_path, existing_files=existing_files, word2vec_model=word2vec_model,  tfidf_scores_dict=tfidf_scores_dict, mdic1=mdic1, mdic2=mdic2, flag2=flag2), dotfiles)






if __name__ == '__main__':
    main()
