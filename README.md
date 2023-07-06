# 多模态情感分析
这是一个实现为给定配对的文本和图像，预测对应的情感标签的双流模型。

## 实验环境配置
```
python 3.8.10
```

```
torch==1.12.1
numpy==1.23.5
PIL==9.5.0
sklearn==1.2.2
transformers==4.30.2
matplotlib==3.7.1
```
可以运行以下命令进行配置：
```
pip install -r requirements.txt
```

## 文件结构
```
|-- large-scale # experiments for 6 large-scale datasets
    |-- data/ # some large-scale datasets
    |-- dataset/  # the remaining large-scale datasets
    |-- experiments/  # all run shs
    |-- main.py # the main code
    |-- main_z.py # obtains coefficient matrix z 
    |-- models.py # includes all model implementations
|-- paper-plots # all experimental plots in our paper
|-- small-scale # experiments for 9 small-scale datasets
    |-- data/ # 3 old datasets, including cora, citeseer, and pubmed
    |-- new-data/ # 6 new datasets, including texas, wisconsin, cornell, actor, squirrel, and chameleon
    |-- splits/ # splits for 6 new datasets
    |-- sh/ # all run shs
    |-- main.py  # the main code
    |-- main_z.py  # obtains coefficient matrix z
    |-- main_h.py # obtains final layer embedding h
```
