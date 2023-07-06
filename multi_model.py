import torch
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
# from tokens import SpecialTokens
from transformers import BertTokenizer
import torch.optim as optim
import torch.cuda as cuda
import matplotlib.pyplot as plt
from model import DualStreamModel
import random
import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')
# 添加参数
parser.add_argument('--mode', type=str, help='choose a mode: img, txt, all', required=True)
parser.add_argument('--data_aug', type=bool, help='if data_augmentation: True or False', required=True)

# 解析命令行参数
args = parser.parse_args()

# 在GPU上运行
torch.cuda.empty_cache()
device = 'cuda' if cuda.is_available() else 'cpu'
# device = 'cpu'

# 分词器 
tokenizer = BertTokenizer.from_pretrained(r'tokenizer')

data_max = 5130

def one_hot(y):
    if y == 'positive':
        return 0
    elif y == 'neutral':
        return 1
    elif y == 'negative':
        return 2
    
def label(y):
    if y == 0:
        return 'positive'
    elif y == 1:
        return 'neutral'
    elif y == 2:
        return 'negative'

# 读取有标签数据的代号和对应标签
def data_index_load(path):
    dataset = []
    with open(path + 'train.txt') as f:
        # 去掉第一行
        f.readline()
        content = f.readlines()
        f.close()

    content = [x.strip() for x in content]  # 去除每行首尾空格
    data = np.array([x.split(',') for x in content])
    index = np.array([int(i.split('.')[0]) for i in data[:,0]])
    
    dataset.append(index)
    dataset.append(np.array([one_hot(i) for i in data[:,1]]))

    with open(path + 'test_without_label.txt') as f:
        # 去掉第一行
        f.readline()
        content = f.readlines()
        f.close()

    content = [x.strip() for x in content]  # 去除每行首尾空格
    data = np.array([x.split(',') for x in content])
    index = np.array([int(i.split('.')[0]) for i in data[:,0]])
    dataset.append(index)

        # print(dataset[i][:,0])
    return dataset[0], dataset[1], dataset[2]


# 加载JPEG图像，并进行预处理
def load_and_preprocess_image(image_path):
    # 指定CNN输入图像的大小
    input_size = (256, 256)
    # 使用PIL库加载JPEG图像
    image = Image.open(image_path)
    # 尺寸调整
    image = image.resize(input_size)
    # 将图像转换为NumPy数组
    image_array = np.array(image)
    # 归一化
    normalized_image_array = (image_array.astype(float) / 255.0)
    # 添加批次维度
    image_tensor = torch.from_numpy(np.transpose(normalized_image_array, (2, 0, 1)))
    # print(image_tensor.shape) #(3,256,256)
    return image_tensor

# 读取所有图像和文本数据，以字典形式存储
def data_load(path):
    dataset = {}
    dataset['jpg'] = []
    dataset['txt'] = []
    # 0位是空列表
    dataset['jpg'].append([])
    dataset['txt'].append('')
    path += 'data/'
    for i in range(1, data_max):
        # 文件不存在就插入空列表
        dataset['jpg'].append([])
        dataset['txt'].append('')
        img_path = os.path.join(path, str(i) + '.jpg')
        txt_path = os.path.join(path, str(i) + '.txt')
        if os.path.exists(img_path):
            image_tensor = load_and_preprocess_image(img_path)
            dataset['jpg'][i] = image_tensor
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='gb18030') as f:
                # file_content = f.readline().split(' ')
                dataset['txt'][i] = f.readline()
            f.close()
    dataset['txt'] = tokenizer(dataset['txt'], padding = True)['input_ids']
    # print(dataset['txt']['input_ids'])
    return dataset

# dataloader封装
def dataloader(dataset, index, label, batch_size):
    img = []
    txt = []
    num_parts = index.shape[0] // batch_size
    batches = np.split(index[:num_parts*batch_size], num_parts)
    batches_label = np.split(label[:num_parts*batch_size], num_parts)

    batches.append(index[num_parts*batch_size:])
    if len(label[num_parts*batch_size:]) != 0:
        batches_label.append(label[num_parts*batch_size:])
        num_parts += 1
    
    # print(num_parts)
    data_loader = []
    # print('here')
    for i in range(num_parts):
        img.append([dataset['jpg'][key] for key in batches[i]])
        txt.append([dataset['txt'][key] for key in batches[i]])
        data_loader.append((batches[i], img[i], txt[i], batches_label[i])) #（序号， 图像， 文本， 标签）
        # print(batches[i])
    # print(num_parts)
    
    return data_loader
    
def data_aug_dataloader(dataset, x, label, batch_size, if_over = False):
    # print('enter')
    if if_over:
        ori = len(x)
        for i in range(ori):
            dataset['jpg'].append([])
            dataset['txt'].append([0])
            # 过采样处理，翻倍
            if label[i] == 1 or label[i] == 2:
                # print('````')
                # 在数据末尾添加 index = data_max+i 的数据
                x = np.append(x, data_max + i)
                dataset['jpg'][data_max + i] = torch.flip(dataset['jpg'][x[i]], dims=(2,))
                # 对文字处理是交换位置
                tmp_list = dataset['txt'][x[i]].copy()
                
                tmp = tmp_list[2]
                tmp_list[2] = tmp_list[3]
                tmp_list[3] = tmp
                # print('`````')
                # print(dataset['txt'][x[i]])
                # print(tmp_list)
                dataset['txt'][data_max + i] = tmp_list
                # print(x[i])
                label = np.append(label, label[i])
        # print(len(x)) # 4489 - 3200
        # print(np.where(x == 5300))
        # print(label[np.where(x == 5300)])
        random.seed(0)
        random.shuffle(x)
        random.seed(0)
        random.shuffle(label)
        # print(np.where(x == 5300))
        # print(label[np.where(x == 5300)])
    train_data = dataloader(dataset, x, label, batch_size)
                
    return train_data                

# 评价
def evaluate_accuracy(val_data, net):
    item_sum = 0
    item_acc = 0
    for x, x_img, x_txt, y in val_data:
        img = torch.tensor([item.numpy() for item in x_img]).float().to(device)
        # img = torch.tensor(np.array(x_img)).float().to(device) 
        txt = torch.from_numpy(np.array(x_txt)).to(device)
        y = torch.tensor(y).to(device)
        # print(x)
        # print(txt)
        y_hat = net(img, txt)
        # print(y_hat.argmax(dim=1))
        item_sum += y.shape[0]
        item_acc += (y_hat.argmax(dim=1) == y).sum().item()
        # print('```````````````````')
        # print(y_hat.argmax(dim=1))
        # print(y)
    return item_acc / item_sum

# train
def train(net, train_data, val_data, image_optimizer, text_optimizer, num_epochs):

    epoch_x = []  # 保存等会更新的epoch_x，loss_y, train_acc, test_acc，用于绘制折线图
    loss_y = []
    acc_train_y = []
    acc_test_y = []
    # 开始训练
#     loss_func = torch.nn.CrossEntropyLoss()
    
    loss_func = torch.nn.CrossEntropyLoss()
    f = open('data_record.txt','w',encoding = 'utf-8')
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count= 0.0, 0.0, 0, 0 # 初始化参数
        for x, x_img, x_txt, y in train_data:
            # 提交到GPU
            # print(x_img)
            # print(batch_count)
            x_img = torch.tensor([item.numpy() for item in x_img]).float().to(device)
            # x_img = torch.tensor(np.array(x_img)).float().to(device) 

            # print(x_txt)
            x_txt = torch.from_numpy(np.array(x_txt)).to(device)
            # print(x_img.shape)
            # print(x_txt.shape)
            y = torch.tensor(y).to(device)

            # 预测
            y_hat = net(x_img, x_txt)
            # print(y_hat)
            loss = loss_func(y_hat, y)  # 使用交叉熵计算loss
            # optimizer.zero_grad()
            # loss.backward()   # 反向传播
            # optimizer.step()
            image_optimizer.zero_grad()
            text_optimizer.zero_grad()
            
            # 反向传播和参数更新
            loss.backward()
            image_optimizer.step()
            text_optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(val_data, net)  # 测试当个epoch的训练的网络
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc))
        epoch_x.append(epoch + 1)  # 追加坐标值
        loss_y.append(train_loss_sum / batch_count)
        acc_train_y.append(train_acc_sum / n)
        acc_test_y.append(test_acc)
    # 绘图
    plt.plot(epoch_x, loss_y, color = 'red')
    plt.plot(epoch_x, acc_train_y, color = 'blue')
    plt.plot(epoch_x, acc_test_y, color = 'yellow')
    plt.ylabel("epoch")
    plt.plot(epoch_x, loss_y, label="loss")  # 在绘图函数添加一个属性label
    plt.plot(epoch_x, acc_train_y, label="train_acc")
    plt.plot(epoch_x, acc_test_y, label="test_acc")
    plt.legend(loc=1)  # 添加图例
    plt.grid()   # 添加网格
    plt.savefig("result.jpg")
    plt.show()  # 显示图片
    return net

def predict(test_x, dataset, net):
    f = open('data/test_without_label.txt', 'w', encoding='utf-8')
    f.write('guid,tag' + '\n')
    for i in test_x:
        img = torch.tensor([dataset['jpg'][i].numpy()]).float().to(device)
        txt = torch.from_numpy(np.array([dataset['txt'][i]])).to(device)
        # print(img.shape)
        # print(txt.shape)
        # print(net(img, txt))
        # print(net(img, txt).argmax(dim = 1))
        
        # print(net(img, txt).argmax(dim = 1)[0])
        result = label(net(img, txt).argmax(dim = 1)[0])
        f.write(str(i) + ',' + result + '\n')
    f.close

path = 'data/'
# 导入index
train_x, train_y, test_x = data_index_load(path)
# print(train_data)
# print(train_x.shape) #（3200,）
# print(val_y.shape) #(800,)

# 划分训练集与验证集
train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.2,random_state=5)
# print(len(train_x))

# 导入图像和文本数据
dataset = data_load(path)
# dataloader封装
val_data = dataloader(dataset, val_x, val_y, batch_size = 256)
train_data = data_aug_dataloader(dataset, train_x, train_y, batch_size = 256, if_over = args.data_aug)
# train_data = dataloader(dataset, train_x, train_y, batch_size = 256)


# 训练(mode = 'img'/'txt'/'all', 默认all)
model = DualStreamModel(max(max(row) for row in dataset['txt']) + 1, mode = args.mode)
model = model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=0.001) # , momentum=0.9
image_optimizer = torch.optim.Adam(model.image_model.parameters(), lr=0.00001)
text_optimizer = torch.optim.Adam(model.text_model.parameters(), lr=0.0001)
num_epochs = 20
# net = train(model, train_data, val_data, optimizer, num_epochs)
net = train(model, train_data, val_data, image_optimizer, text_optimizer, num_epochs)


predict(test_x, dataset, net)
torch.cuda.empty_cache()