import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os
import jieba

def load_data(file_path, sample_size=100000):
    """
    加载并解析数据集，并且限制数据量
    文件内的数据格式为: ID_!_分类ID_!_一级分类_!_标题_!_关键词
    """
    data = []
    if not os.path.exists(file_path):
        print(f"错误：数据文件 '{file_path}' 未找到。")
        return None 

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size is not None and i >= sample_size:
                    break
                parts = line.strip().split('_!_')
                if len(parts) >= 4:
                    content = parts[3]  # 标题作为主要内容
                    primary_category = parts[2].replace('news_', '')  # 一级分类
                    keywords = parts[4] if len(parts) > 4 else ""  # 关键词

                    # 将关键词加入内容中增强特征
                    full_content = content + " " + keywords

                    data.append({
                        'content': full_content,
                        'primary_category': primary_category
                    })
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

    if not data:
        print("未能从文件中加载任何数据。")
        return None

    return pd.DataFrame(data)

def load_stopwords(filepath='stopwords.txt'):
    """加载停用词列表并进行分词处理，确保与tokenizer一致"""
    stopwords = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # 对每个停用词进行分词，确保与chinese_tokenizer一致
                    word = line.strip()
                    if word:
                        # 将原始停用词添加到集合
                        stopwords.add(word)
                        # 对停用词进行分词，将分词结果也添加到停用词集合
                        for token in jieba.cut(word):
                            if token.strip():
                                stopwords.add(token.strip())
            print(f"成功加载并处理 {len(stopwords)} 个停用词。")
        except Exception as e:
            print(f"加载停用词时出错: {e}")
    else:
        print(f"警告：停用词文件 '{filepath}' 未找到，将不使用停用词。")
    return list(stopwords) if stopwords else None

def chinese_tokenizer(text):
    """使用 jieba 分词"""
    return list(jieba.cut(text))

def train_and_evaluate(data_path='toutiao_news_data.txt', sample_size=20000, model_path='article_classifier.pkl', stopwords_path='stopwords.txt'):
    """
    加载数据、训练模型、评估并保存模型。
    """
    # 加载数据
    df = load_data(data_path, sample_size=sample_size)
    if df is None or df.empty:
        print("数据加载失败或数据为空，无法继续训练。")
        return None, None 

    # 对类别标签进行编码
    le = LabelEncoder()
    try:
        df['primary_category'] = le.fit_transform(df['primary_category'])
    except Exception as e:
        print(f"标签编码时出错: {e}")
        return None, None

    # 分割训练集和测试集
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df['content'], df['primary_category'], test_size=0.3, random_state=42, stratify=df['primary_category']
        )
    except Exception as e:
        print(f"数据分割时出错: {e}")
        return None, None

    # 加载停用词
    stopwords = load_stopwords(stopwords_path)

    # 定义模型管道
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=chinese_tokenizer, # 使用 jieba 分词
            stop_words=stopwords if stopwords else None, # 应用停用词
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            token_pattern=None, # 设置为None以避免警告，因为使用了自定义tokenizer
            max_features=10000)),
        ('clf', LinearSVC(C=1.0, dual=False, class_weight='balanced'))
    ])

    print("开始训练 SVM 模型...")
    try:
        model.fit(X_train, y_train)
        print("模型训练完成。")
    except Exception as e:
        print(f"模型训练时出错: {e}")
        return None, None

    # 
    print("\n开始评估模型...")
    try:
        y_pred = model.predict(X_test)
        print("\n分类报告:")
        target_names = le.classes_
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"模型评估时出错: {e}")


    model_data = {
        'model': model,
        'label_encoder': le
    }
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型和标签编码器已保存到 {model_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        return None, None

    return model, le

def classify_article_loaded(text, loaded_model, loaded_label_encoder):
    """
    使用已加载的模型和标签编码器对输入的文章内容进行分类。
    参数:
        text: 要分类的文本
        loaded_model: 已加载的 Pipeline 模型
        loaded_label_encoder: 已加载的 LabelEncoder
    返回: 一级分类名称
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "错误：输入文本无效"

    try:
        prediction = loaded_model.predict([text])[0]
        category = loaded_label_encoder.inverse_transform([prediction])[0]
        return category
    except Exception as e:
        print(f"分类时出错: {e}")
        if "not fitted" in str(e).lower():
             return "错误：模型或其组件未正确加载或训练。"
        elif "dimension mismatch" in str(e).lower():
             return "错误：输入数据的特征维度与模型训练时不符。"
        else:
             return f"错误：分类过程中发生未知错误 ({type(e).__name__})"


if __name__ == "__main__":
    # 训练、评估并保存模型
    trained_model, label_encoder = train_and_evaluate(sample_size=10000, stopwords_path='stopwords.txt')

    # 检查模型是否成功训练
    if trained_model and label_encoder:
        print("\n开始测试分类函数...")
        test_text = '量子计算机取得重大突破，未来计算能力将指数级增长'
        predicted_category = classify_article_loaded(test_text, trained_model, label_encoder)
        print(f"测试文章: '{test_text}'")
        print(f"预测分类: {predicted_category}")

        test_text_2 = '下赛季的中超联赛将迎来新的赞助商'
        predicted_category_2 = classify_article_loaded(test_text_2, trained_model, label_encoder)
        print(f"测试文章: '{test_text_2}'")
        print(f"预测分类: {predicted_category_2}")
    else:
        print("\n模型训练或保存失败，无法进行测试分类。")