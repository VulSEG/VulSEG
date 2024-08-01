import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer


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
        print(filename)
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


def write_dic():
    directory_path = "./corpus"
    tfidf_scores_dict = calculate_tfidf_for_corpus(directory_path)
    with open('./dic2.txt', 'w') as file:
        json.dump(tfidf_scores_dict, file)

if __name__ == '__main__':
    write_dic()