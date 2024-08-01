import pickle, os, glob
import argparse
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
# 将数据保存到指定的文件中
def sava_data(filename, data):
    print("开始保存数据至于：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

# 从指定的文件中加载数据
def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

# 解析命令行参数
def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split train datasettest_data.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some pkl_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-n', '--num', help='Num of K-fold.', required=True)
    args = parser.parse_args()
    return args

# 从指定路径生成DataFrame
def generate_dataframe(input_path, save_path):
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    print(input_path)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dic = []

    i = 0
    for type_name in os.listdir(input_path):

        dicname = input_path + type_name
        print(dicname)
        filename = glob.glob(dicname + "/*.pkl")
        for file in filename:
            file2 = file.replace('/all_pdg2/', '/all_cfg2/')
            data = load_data(file)
            try:
                data_2 = load_data(file2)
            except Exception as e:
                print("not in ************************")
                data_2 = load_data(file)
            data1 = [data[0], data[1], data[2]]
            data2 = [data[3], data[4], data[5]]
            data3 = [data_2[0], data_2[1], data_2[2]]
            dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"), 
                "length1":   len(data[0]),
                "length2": len(data[3]),
                "length3": len(data_2[0]),
                "data1": data1,
                "data2": data2,
                "data3": data3,
                "label":    0 if type_name == "No-Vul" else 1})
            i = i + 1
            print(i)
            print(type_name)



    final_dic = pd.DataFrame(dic)
    sava_data(save_path + "all_data.pkl", final_dic)

def gather_data(input_path, output_path):
    generate_dataframe(input_path, output_path)

# 将所有数据划分为训练集和测试集，根据K折交叉验证
def split_data(all_data_path, save_path, kfold_num):
    kfold_num = int(kfold_num)
    df_test = load_data(all_data_path)
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    seed = 0
    df_dict = {}
    train_dict = {i:{} for i in range(kfold_num)}
    test_dict = {i:{} for i in range(kfold_num)}

    kf = KFold(n_splits = kfold_num, shuffle = True, random_state = seed)
    for i in Counter(df_test.label.values):
        df_dict[i] = df_test[df_test.label == i]
        for epoch, result in enumerate(kf.split(df_dict[i])):
            train_dict[epoch][i]  = df_dict[i].iloc[result[0]]
            test_dict[epoch][i] =  df_dict[i].iloc[result[1]] 
    train_all = {i:pd.concat(train_dict[i], axis=0, ignore_index=True) for i in train_dict}
    test_all = {i:pd.concat(test_dict[i], axis=0, ignore_index=True) for i in test_dict}
    sava_data(save_path + "train.pkl", train_all)
    sava_data(save_path + "test.pkl", test_all)

def main():
    args = parse_options()
    input_path = args.input
    output_path = args.out
    kfold_num = args.num
    gather_data(input_path, output_path)

    split_data(output_path + "/all_data.pkl", output_path, kfold_num)
    

if __name__ == "__main__":
    main()