import argparse
from model import load_data, CNN_Classifier

# 解析命令行参数
def parse_options():
    parser = argparse.ArgumentParser(description='VulSEG training.')
    parser.add_argument('-i', '--input', help='The dir path of train.pkl and test.pkl', type=str, required=True)
    args = parser.parse_args()
    return args

# 获取K折交叉验证的DataFrame
def get_kfold_dataframe(pathname = "./data/", item_num = 0):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")[item_num]  # 加载训练数据
    eval_df = load_data(pathname + "test.pkl")[item_num]    # 加载测试/验证数据

    return train_df, eval_df    # 返回训练和测试/验证DataFrame

def main():
    args = parse_options()
    item_num = 0
    hidden_size1 = 128
    hidden_size2 = 100
    hidden_size3 = 128
    data_path = args.input
    for item_num in range(5):
        train_df, eval_df = get_kfold_dataframe(pathname = data_path, item_num = item_num)
        classifier = CNN_Classifier(result_save_path = data_path.replace("pkl", "results"), \
            item_num = item_num, epochs=2000, hidden_size1 = hidden_size1,  hidden_size2 = hidden_size2, hidden_size3 = hidden_size3)
        print(train_df)
        classifier.preparation(
            X_train1=train_df['data1'],
            X_train2=train_df['data2'],
            X_train3=train_df['data3'],
            y_train=train_df['label'],
            X_valid1=eval_df['data1'],
            X_valid2=eval_df['data2'],
            X_valid3=eval_df['data3'],
            y_valid=eval_df['label']
        )
        classifier.train()


if __name__ == "__main__":
    main()