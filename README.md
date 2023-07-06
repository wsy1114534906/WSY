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
|-- tokenizer # 分词器依赖的配置文件
    |-- config.json
    |-- vocab.txt
|-- model.py # 模型代码
|-- multi_model.py # 其他函数代码
|-- README.md
|-- requirements.txt # 代码执行环境
|-- data # 数据文件
    |-- __MACOSX
    |-- data # 图片、文本
    |-- test_without_label.txt
    |-- train.txt
|-- result # 实验结果文件
|-- 实验5.doc # 实验报告
```

## 代码执行
命令行输入 (all/img/txt)(True/False)
```
python multi_model.py –mode all –data_aug True
```

## 参考
实验三、深度学习期末作业
