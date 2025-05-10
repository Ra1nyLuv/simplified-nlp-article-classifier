import numpy as np
import pandas as pd
import pickle
import time
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder

# --- 从 article_classifier.py 引入辅助函数 ---
def load_stopwords(filepath='stopwords.txt'):
    """加载停用词列表并进行分词处理，确保与tokenizer一致"""
    stopwords = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # 对每个停用词进行处理
                    word = line.strip()
                    if word:
                        # 将原始停用词添加到集合
                        stopwords.add(word)
                        
                        # 对停用词进行分词，将分词结果也添加到停用词集合
                        for token in jieba.cut(word):
                            if token.strip():
                                stopwords.add(token.strip())
                        
                        # 处理特殊字符和标点符号
                        for char in word:
                            if char.strip():
                                stopwords.add(char.strip())
                        
                        # 处理可能的Unicode字符
                        for char in word:
                            if r'\u' in repr(char):
                                stopwords.add(char)
            
            # 添加常见的特殊字符和标点符号，这些在警告中可能出现
            special_chars = ['lex', 'δ', 'ψ', 'в', 'ⅲ', 'ｌ', 'ｒ', 'ｔ', 'ｘ', 'ｚ']
            for char in special_chars:
                stopwords.add(char)
                
            # 添加所有单个字母，解决警告中提到的'l'不在停用词中的问题
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                stopwords.add(letter)
                
            print(f"成功加载并处理 {len(stopwords)} 个停用词。")
        except Exception as e:
            print(f"加载停用词时出错: {e}")
    else:
        print(f"警告：停用词文件 '{filepath}' 未找到，将不使用停用词。")
    return list(stopwords) if stopwords else None # 返回 list 或 None

def chinese_tokenizer(text):
    """使用 jieba 分词"""
    return list(jieba.cut(text))


def load_data(file_path, sample_size=None): 
    """
    加载并解析数据集，允许采样
    格式: ID_!_分类ID_!_一级分类_!_标题_!_关键词
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
                    content = parts[3]  # 将标题作为主要内容
                    primary_category = parts[2].replace('news_', '')  # 一级分类
                    keywords = parts[4] if len(parts) > 4 else ""  # 关键词

                    # 将关键词加入内容以增强特征
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

def optimize_models(data_path='toutiao_news_data.txt', stopwords_path='stopwords.txt', sample_size=None, save_best=True, cv_folds=3):
    """
    优化模型超参数并比较不同模型的性能
    """
    print("开始加载数据...")

    df = load_data(data_path, sample_size=sample_size) # 应用采样
    if df is None or df.empty:
        print("数据加载失败或数据为空，优化中止。")
        return

    # 加载停用词
    stopwords = load_stopwords(stopwords_path)

    # 对类别标签进行编码
    le = LabelEncoder()
    try:
        df['primary_category'] = le.fit_transform(df['primary_category'])
    except Exception as e:
        print(f"标签编码时出错: {e}")
        return

    # 分割训练集和测试集 (仍然需要测试集来最终评估优化后的模型)
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['primary_category'], test_size=0.3, random_state=42, stratify=df['primary_category']
    )

    print(f"数据加载完成，共有{len(df)}条数据，{len(le.classes_)}个类别")
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 定义不同的特征提取方法 (集成中文分词和停用词)
    feature_extractors = {
        'tfidf_ngram': TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            stop_words=stopwords,
            ngram_range=(1, 2), # 保持 ngram
            min_df=5, # 使用 article_classifier.py 中调整后的值
            max_df=0.8, # 使用 article_classifier.py 中调整后的值
            max_features=10000, # 使用 article_classifier.py 中调整后的值
            token_pattern=None # 设置为None以避免警告，因为使用了自定义tokenizer
        ),
        # 可以保留或添加其他 vectorizer 变体，确保它们也使用 tokenizer 和 stop_words
        # 'tfidf_basic': TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=stopwords, min_df=5, max_df=0.8, max_features=10000, token_pattern=None),
        # 'count_vector': CountVectorizer(tokenizer=chinese_tokenizer, stop_words=stopwords, min_df=5, max_df=0.8, max_features=10000)
    }

    # 定义不同的分类器 (添加 class_weight='balanced')
    classifiers = {
        'naive_bayes': MultinomialNB(),
        'svm': LinearSVC(class_weight='balanced', dual=False, max_iter=2000),
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', n_jobs=-1), # liblinear 适合二分类和 OvR 多分类
        # 'gradient_boosting': GradientBoostingClassifier(n_estimators=100)
    }

    # 存储结果
    results = []
    best_cv_score = 0
    best_pipeline_components = None # 存储最佳组合的组件名称
    best_initial_pipeline = None # 存储未经GridSearch的最佳pipeline

    # --- 使用交叉验证进行初始模型比较 ---
    print(f"\n--- 开始使用 {cv_folds}-折交叉验证进行初始模型比较 ---")
    for extractor_name, extractor in feature_extractors.items():
        for clf_name, clf in classifiers.items():
            start_time = time.time()

            # 创建管道
            pipeline = Pipeline([
                ('vectorizer', extractor),
                ('classifier', clf)
            ])

            # 执行交叉验证
            print(f"评估模型: {extractor_name} + {clf_name}")
            try:
                # 使用 accuracy 作为评分标准，可以在 make_scorer 中指定其他指标
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1) # 使用多核
                mean_cv_score = np.mean(cv_scores)
                eval_time = time.time() - start_time

                print(f"平均交叉验证准确率: {mean_cv_score:.4f}, 评估时间: {eval_time:.2f}秒")

                results.append({
                    'extractor': extractor_name,
                    'classifier': clf_name,
                    'mean_cv_accuracy': mean_cv_score,
                    'eval_time': eval_time
                })

                # 更新最佳模型 (基于交叉验证分数)
                if mean_cv_score > best_cv_score:
                    best_cv_score = mean_cv_score
                    best_pipeline_components = (extractor_name, clf_name)
                    # 重新构建最佳 pipeline 以便后续 GridSearchCV 使用
                    best_initial_pipeline = Pipeline([
                        ('vectorizer', feature_extractors[extractor_name]),
                        ('classifier', classifiers[clf_name])
                    ])

            except Exception as e:
                eval_time = time.time() - start_time
                print(f"模型 {extractor_name} + {clf_name} 评估出错: {e}, 时间: {eval_time:.2f}秒")
                results.append({
                    'extractor': extractor_name,
                    'classifier': clf_name,
                    'mean_cv_accuracy': 0.0, # 标记为失败
                    'eval_time': eval_time
                })
    # --- 结束交叉验证评估 ---

    # 显示所有结果
    results_df = pd.DataFrame(results)
    print("\n所有模型初始交叉验证性能比较:")
    print(results_df.sort_values('mean_cv_accuracy', ascending=False))

    if best_initial_pipeline is None:
        print("\n未能找到有效的最佳模型组合，无法进行超参数优化。")
        return

    best_extractor_name, best_clf_name = best_pipeline_components
    print(f"\n--- 对最佳组合 {best_extractor_name} + {best_clf_name} (CV Acc: {best_cv_score:.4f}) 进行超参数优化 ---")

    # 根据最佳模型类型设置不同的参数网格
    param_grid = {}
    param_grid.update({
        'vectorizer__min_df': [3, 5, 7],
        'vectorizer__max_df': [0.7, 0.8, 0.9],
        # 'vectorizer__ngram_range': [(1, 1), (1, 2)], # 如果有多个 vectorizer，需要更复杂的逻辑
    })

    if best_clf_name == 'naive_bayes':
        param_grid.update({
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0]
        })
    elif best_clf_name == 'svm':
        param_grid.update({
            'classifier__C': [0.1, 0.5, 1.0, 2.0, 5.0],
            # dual='auto' 通常是较好的选择，或者根据数据特性固定
            # 'classifier__dual': [False], # 如果之前确定 False 更好
        })
    elif best_clf_name == 'random_forest':
        param_grid.update({
            'classifier__n_estimators': [100, 200], # 减少数量以加快速度
            'classifier__max_depth': [None, 20, 50], # 调整深度
            'classifier__min_samples_split': [2, 5]
        })
    elif best_clf_name == 'logistic_regression':
        param_grid.update({
            'classifier__C': [0.1, 1.0, 10.0],
            # 'classifier__solver': ['liblinear', 'saga'] # 如果需要测试不同 solver
        })
    # GradientBoosting 的参数网格可以保持或调整
    # elif best_clf_name == 'gradient_boosting':
    #     param_grid.update({ ... })

    if param_grid:
        print("开始 GridSearchCV...")
        start_time = time.time()
        # 使用网格搜索优化超参数 (在完整的训练集上进行)
        # cv=cv_folds 复用之前的折数设置
        grid_search = GridSearchCV(best_initial_pipeline, param_grid, cv=cv_folds, n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        grid_search_time = time.time() - start_time
        print(f"GridSearchCV 完成，耗时: {grid_search_time:.2f}秒")

        # 获取最佳参数
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"GridSearchCV 最佳交叉验证准确率: {grid_search.best_score_:.4f}")

        # 使用最佳参数评估模型 (在独立的测试集上)
        optimized_model = grid_search.best_estimator_
        y_pred = optimized_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\n在测试集上的最终准确率: {test_accuracy:.4f}")
        print("\n测试集分类报告:")
        # 使用 label_encoder.classes_ 获取类别名称
        target_names = le.classes_
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        # 保存最佳模型
        if save_best:
            model_save_path = 'optimized_classifier.pkl'
            try:
                with open(model_save_path, 'wb') as f:
                    pickle.dump({
                        'model': optimized_model,
                        'label_encoder': le
                    }, f)
                print(f"优化后的模型已保存为 {model_save_path}")
            except Exception as e:
                print(f"保存优化模型时出错: {e}")

    else:
        print("无法为当前最佳模型设置参数网格，将保存交叉验证阶段找到的最佳模型。")
        # 保存交叉验证阶段找到的最佳模型 (未经GridSearch)
        if save_best and best_initial_pipeline:
            model_save_path = 'optimized_classifier.pkl'
            try:
                # 需要在整个训练集上重新训练一次
                print("在整个训练集上重新训练最佳初始模型...")
                best_initial_pipeline.fit(X_train, y_train)
                print("重新训练完成。")

                # 在测试集上评估
                y_pred = best_initial_pipeline.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                print(f"\n在测试集上的最终准确率 (未经GridSearch): {test_accuracy:.4f}")
                print("\n测试集分类报告 (未经GridSearch):")
                target_names = le.classes_
                print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

                with open(model_save_path, 'wb') as f:
                    pickle.dump({
                        'model': best_initial_pipeline,
                        'label_encoder': le
                    }, f)
                print(f"最佳初始模型已保存为 {model_save_path}")
            except Exception as e:
                print(f"保存最佳初始模型时出错: {e}")


if __name__ == "__main__":
    # 可以通过 sample_size 控制用于优化的数据量，None 表示使用全部数据
    optimize_models(sample_size=50000, cv_folds=3)
    print("\n模型优化完成")