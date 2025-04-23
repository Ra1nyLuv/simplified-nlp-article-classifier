# 文章分类系统
## 项目简介
一个基于机器学习的中文新闻文章分类系统，支持多种分类算法和模型优化，提供可视化界面。

## 项目亮点

- 支持SVM、随机森林等多种机器学习算法
- 集成中文分词和停用词过滤
- 提供混淆矩阵和ROC曲线可视化
- 支持模型超参数自动优化
- 基于Streamlit的交互式Web界面

## 功能特性

- 文章自动分类：输入文本实时返回分类结果
- 模型评估：生成分类报告和可视化图表
- 批量测试：支持多文件批量分类测试
- 模型优化：自动寻找最佳参数组合
- 置信度显示：可视化预测置信度进度条

## 技术栈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)

| 组件             | 技术选型                              |
|------------------|-------------------------------------|
| 核心框架         | Scikit-learn, Streamlit             |
| 中文处理         | Jieba分词, 自定义停用词表            |
| 特征工程         | TF-IDF, N-gram                      |
| 机器学习算法     | SVM, Random Forest, Logistic Regression |
| 可视化           | Matplotlib, Seaborn                 |

## 快速开始

### 环境准备

1. 克隆仓库：
```bash
git clone https://github.com/ra1nyluv/nlp-article-classifier.git
cd nlp-article-classifier
```
