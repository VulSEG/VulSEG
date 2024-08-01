
import os
import lap
import torch
import numpy
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


logging.basicConfig(filename='model_metrics.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def sava_data(filename, data):
    print("Begin to save data：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def save_data2(file_path, data):
    with open(file_path, 'w') as f:  # 使用 'w' 参数来覆盖写入
        for epoch, metrics in data.items():
            f.write(f'Epoch {epoch + 1}:\n')
            for key, value in metrics.items():
                if isinstance(value, dict):  # 如果值是字典，进一步遍历
                    f.write(f'{key}:\n')
                    for sub_key, sub_value in value.items():
                        f.write(f'  {sub_key}: {sub_value}\n')
                else:
                    f.write(f'{key}: {value}\n')
            f.write('\n')  # 在每个周期后添加空行

def load_data(filename):
    print("Begin to load data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def get_accuracy(labels, prediction):
    cm = confusion_matrix(labels, prediction)

    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    def linear_assignment(cost_matrix):
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy


def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(labels, predictions, average='macro')
    MCM = multilabel_confusion_matrix(labels, predictions)
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum(sum_array)
    W_fnr = (fnr_array * sum_array).sum() / sum(sum_array)
    W_f1 = (f1_array * sum_array).sum() / sum(sum_array)
    return {
        "M_fpr": format(M_fpr * 100, '.3f'),
        "M_fnr": format(M_fnr * 100, '.3f'),
        "M_f1": format(M_f1 * 100, '.3f'),
        "W_fpr": format(W_fpr * 100, '.3f'),
        "W_fnr": format(W_fnr * 100, '.3f'),
        "W_f1": format(W_f1 * 100, '.3f'),
        "ACC": format(accuracy * 100, '.3f'),
        "MCM": MCM
    }



class TraditionalDataset(Dataset):
    def __init__(self, texts1, texts2, texts3, targets, max_len, hidden_size1, hidden_size2, hidden_size3):
        self.texts1 = texts1
        self.texts2 = texts2
        self.texts3 = texts3
        self.targets = targets
        self.max_len = max_len
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3


    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        feature1 = self.texts1[idx]
        feature2 = self.texts2[idx]
        feature3 = self.texts3[idx]
        target = self.targets[idx]
        vectors1 = numpy.zeros(shape=(3, self.max_len, self.hidden_size1))
        vectors2 = numpy.zeros(shape=(3, self.max_len, self.hidden_size2))
        vectors3 = numpy.zeros(shape=(3, self.max_len, self.hidden_size3))
        for j in range(3):
            for i in range(min(len(feature1[0]), self.max_len)):
                vectors1[j][i] = feature1[j][i]
            for i in range(min(len(feature2[0]), self.max_len)):
                vectors2[j][i] = feature2[j][i]
            for i in range(min(len(feature3[0]), self.max_len)):
                vectors3[j][i] = feature3[j][i]
        return {
            'vector1': vectors1,
            'vector2': vectors2,
            'vector3': vectors3,
            'targets': torch.tensor(target, dtype=torch.long)
        }


class TextCNN(nn.Module):

    def __init__(self, hidden_size1, hidden_size2, hidden_size3, num_classes=2, rnn_hidden1=128, rnn_hidden2=100, rnn_hidden3=128):
        super(TextCNN, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.num_filters = 32
        classifier_dropout = 0.3

        # 定义第一种输入的卷积层
        self.convs1 = nn.ModuleList([
            nn.Conv2d(3, self.num_filters, (k, hidden_size1)) for k in self.filter_sizes
        ])
        # 定义第二种输入的卷积层
        self.convs2 = nn.ModuleList([
            nn.Conv2d(3, self.num_filters, (k, hidden_size2)) for k in self.filter_sizes
        ])

        self.convs3 = nn.ModuleList([
            nn.Conv2d(3, self.num_filters, (k, hidden_size3)) for k in self.filter_sizes
        ])

        # 第一种特征向量的BiLSTM层
        self.lstm1 = nn.LSTM(input_size=self.num_filters * len(self.filter_sizes),
                             hidden_size=rnn_hidden1,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)

        # 第二种特征向量的BiLSTM层
        self.lstm2 = nn.LSTM(input_size=self.num_filters * len(self.filter_sizes),
                             hidden_size=rnn_hidden2,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)

        self.lstm3 = nn.LSTM(input_size=self.num_filters * len(self.filter_sizes),
                             hidden_size=rnn_hidden3,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)
        # Dropout层
        self.dropout = nn.Dropout(classifier_dropout)

        # 全连接层
        self.fc = nn.Linear(rnn_hidden1 * 2 + rnn_hidden2 * 2 + rnn_hidden3 * 2, num_classes)  # 计算每个BiLSTM的双向输出总和

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x1, x2, x3):
        out1 = x1.float()
        out2 = x2.float()
        out3 = x3.float()


        # 对第一种输入应用卷积和池化
        hidden_state1 = torch.cat([self.conv_and_pool(out1, conv) for conv in self.convs1], 1)
        hidden_state1 = hidden_state1.unsqueeze(1)  # 增加batch维度

        # 对第二种输入应用卷积和池化
        hidden_state2 = torch.cat([self.conv_and_pool(out2, conv) for conv in self.convs2], 1)
        hidden_state2 = hidden_state2.unsqueeze(1)  # 增加batch维度

        hidden_state3 = torch.cat([self.conv_and_pool(out3, conv) for conv in self.convs3], 1)
        hidden_state3 = hidden_state3.unsqueeze(1)  # 增加b
        # 应用第一个BiLSTM
        lstm_out1, _ = self.lstm1(hidden_state1)
        lstm_out1 = lstm_out1[:, -1, :]  # 取第一个BiLSTM最后的输出

        # 应用第二个BiLSTM
        lstm_out2, _ = self.lstm2(hidden_state2)
        lstm_out2 = lstm_out2[:, -1, :]  # 取第二个BiLSTM最后的输出


        lstm_out3, _ = self.lstm3(hidden_state3)
        lstm_out3 = lstm_out3[:, -1, :]  # 取第二个BiLSTM最后的输出

        # 拼接两个BiLSTM的输出
        out = torch.cat((lstm_out1, lstm_out2, lstm_out3), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def save_checkpoint(state, filename="model_checkpoint.pth"):
    print("保存模型")
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    print("加载模型")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

class CNN_Classifier():
    def __init__(self, max_len=200, n_classes=2, epochs=100, batch_size=128, learning_rate=0.001, \
                 result_save_path="/root/data/qm_data/results", item_num=0, hidden_size1=128, hidden_size2=100, hidden_size3=128):
        self.model = TextCNN(hidden_size1, hidden_size2, hidden_size3)
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        # torch.cuda.set_device(2)
        result_save_path = result_save_path + "/" if result_save_path[-1] != "/" else result_save_path
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)
        self.result_save_path = result_save_path + str(item_num) + "_epo" + str(epochs) + "_bat" + str(
            batch_size) + ".result"

    def preparation(self, X_train1, X_train2, X_train3, y_train, X_valid1, X_valid2, X_valid3, y_valid):
        # create datasets
        self.train_set = TraditionalDataset(X_train1, X_train2, X_train3, y_train, self.max_len, self.hidden_size1, self.hidden_size2, self.hidden_size3)
        self.valid_set = TraditionalDataset(X_valid1, X_valid2, X_valid3, y_valid, self.max_len, self.hidden_size1, self.hidden_size2, self.hidden_size3)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vector1 = data["vector1"].to(self.device)
            vector2 = data["vector2"].to(self.device)
            vector3 = data["vector3"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(vector1, vector2, vector3)
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()

            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))
            labels += list(np.array(targets.cpu()))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for _, data in progress_bar:
                vector1 = data["vector1"].to(self.device)
                vector2 = data["vector2"].to(self.device)
                vector3 = data["vector3"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs = self.model(vector1, vector2, vector3)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))

                losses.append(loss.item())
                progress_bar.set_description(
                    f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets) / len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ", val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    def train(self, resume=False, checkpoint_path=None):
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])

        resume = True
        checkpoint_path = 'checkpoint_epoch_modeln2.pth'
        if resume and checkpoint_path:
            start_epoch = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        else:
            start_epoch = 0
        i = 0
        for epoch in range(start_epoch, self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(
                ["tra", str(epoch + 1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if
                                                                      j != "MCM"])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(
                ["val", str(epoch + 1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                                           "train_score": train_score, "val_score": val_score}

            # sava_data(self.result_save_path, learning_record_dict)
            if i % 10 == 0:
                save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'loss': train_loss
                    }, 'checkpoint_epoch_model.pth')
            i = i + 1
            print("\n")