import os
import glob
from gensim.models import Word2Vec

def read_texts(directory_path):
    all_texts = []
    for filepath in glob.glob(os.path.join(directory_path, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as file:
            # 读取文件内容并以空格分割
            text = file.read().strip().split(' ')
            all_texts.append(text)
    return all_texts





def train_word2vec(texts):
    # 训练 Word2Vec 模型
    model = Word2Vec(sentences=texts, vector_size=128, window=5, min_count=1, workers=4)
    # 保存模型以便以后使用
    model.save("./model/word2vec2_model3.model")
    return model


if __name__ == '__main__':
    directory_path = './corpus'
    texts = read_texts(directory_path)
    # 训练模型
    model = train_word2vec(texts)

