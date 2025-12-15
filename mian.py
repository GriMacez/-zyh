# 导入基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from collections import Counter

# 图像处理库 (PIL/OpenCV)
from PIL import Image

# NLP 处理库
import jieba

# Scikit-learn 用于数据划分和CV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix

# PyTorch (用于模型和CV循环)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 设置中文显示（解决图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows可用
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac可用
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
RANDOM_SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(RANDOM_SEED)