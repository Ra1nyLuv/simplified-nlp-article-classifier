import streamlit as st
import numpy as np
import pickle
import os
import jieba 
from article_classifier import chinese_tokenizer #
# import io 

# --- 使用缓存装饰器 ---
@st.cache_resource # 使用 cache_resource 更适合加载模型等资源
def load_model(model_path='article_classifier.pkl'):
    # 优先尝试加载优化后的模型
    optimized_model_path = 'optimized_classifier.pkl'
    load_path = model_path # 默认加载基础模型
    if os.path.exists(optimized_model_path):
        load_path = optimized_model_path
        print(f"发现优化后的模型，将加载: {load_path}")
    elif not os.path.exists(model_path):
         raise FileNotFoundError(f"模型文件 '{model_path}' 和 '{optimized_model_path}' 都不存在。")
    else:
        print(f"未找到优化后的模型，将加载基础模型: {load_path}")

    try:
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        # 引发异常，让调用者处理 Streamlit 错误
        raise FileNotFoundError(f"模型文件 '{load_path}' 未找到。")
    except Exception as e:
        # 引发异常
        raise RuntimeError(f"加载模型 '{load_path}' 时出错: {e}")

    # 检查加载的数据是否包含必要的键
    if 'model' not in model_data or 'label_encoder' not in model_data:
        # 引发异常
        raise ValueError(f"模型文件 '{load_path}' 缺少 'model' 或 'label_encoder'。")

    return model_data['model'], model_data['label_encoder']

# 分类函数 
def classify_article(text, model, label_encoder):
    if not text.strip():
        return "请输入文章内容", None

    # 使用加载的 Pipeline 模型直接预测 (它内部包含向量化)
    try:
        # text 需要是可迭代的，所以传入 [text]
        prediction = model.predict([text])[0]
    except Exception as e:
        st.error(f"模型预测失败: {e}")
        return "预测错误", None

    # 解码类别
    try:
        category = label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        st.error(f"类别解码失败: {e}")
        return "解码错误", None

    # 获取预测概率 (如果分类器支持)
    confidence = None
    # Pipeline 的最后一步是分类器，检查它是否支持 predict_proba
    final_estimator = model.steps[-1][1]
    if hasattr(final_estimator, 'predict_proba'):
        try:
            # 同样，需要传入 [text]
            # 使用 transform 获取特征向量，然后传递给 predict_proba
            text_vector = model.transform([text])
            proba = final_estimator.predict_proba(text_vector)[0]
            max_proba = max(proba) * 100
            confidence = max_proba
        except Exception as e:
            # 如果获取概率出错，仍然返回类别，但置信度为 None
            print(f"获取置信度时出错: {e}")
            pass # confidence 保持为 None
    else:
         print("模型 (或其最终分类器) 不支持 predict_proba")


    return category, confidence


def main():
    st.set_page_config(page_title="文章分类系统", layout="wide")

    # 页面标题
    st.title("文章分类系统")

    # 加载模型 (移到 try-except 块中处理错误)
    try:
        model, label_encoder = load_model()
    except FileNotFoundError as e:
        st.error(str(e) + " 请先运行 article_classifier.py 或 model_optimization.py。")
        st.stop()
    except (RuntimeError, ValueError) as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"模型加载过程中发生意外错误: {e}")
        st.stop()


    # 初始化 session_state 以存储文本内容和分类结果
    if 'text_for_classification' not in st.session_state:
        st.session_state.text_for_classification = ""
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None


    # 使用表单实现回车提交
    with st.form(key='classification_form'):
        # 文本输入区域，其状态由 st.session_state.text_for_classification 管理
        # The key "text_for_classification" ensures st.session_state.text_for_classification is updated
        # with the text_area's content upon interaction.
        # The value argument sets the initial display value.
        st.text_area("请输入或粘贴文章内容:", value=st.session_state.text_for_classification, height=200, key="text_for_classification")
        # 表单提交按钮
        submitted = st.form_submit_button("开始分类 (或在上方文本框按 Enter)")

        if submitted:
            # 直接从 session_state 获取文本内容，该状态已由 text_area 通过 key 更新。
            content_to_classify = st.session_state.text_for_classification
            # 此处不再需要 st.session_state.text_for_classification = content_to_classify，因为它是多余的。

            if content_to_classify.strip():
                with st.spinner("正在进行分类..."):
                    category, confidence = classify_article(content_to_classify, model, label_encoder)
                    st.session_state.classification_result = (category, confidence)
            else:
                st.warning("请输入文章内容")
                st.session_state.classification_result = None # 清除旧结果


    # 在表单外部显示结果
    st.subheader("分类结果:")
    if st.session_state.classification_result:
        category, confidence = st.session_state.classification_result
        if category.endswith("错误") or category == "请输入文章内容":
             st.error(category)
        elif confidence is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"类别: {category}")
            with col2:
                st.info(f"置信度: {confidence:.2f}%")
            st.progress(confidence / 100)
        else:
            st.info(f"类别: {category}")
    else:
        st.write("等待输入文本进行分类...") # 修改提示


    # 添加说明信息
    with st.expander("使用说明"):
        st.markdown("""
        1.  **输入文本**: 在文本框中输入或粘贴文章内容。
        2.  **开始分类**:
            *   输入文本后，可以按下**Ctrl+Enter**提交。
            *   或者，点击 **开始分类** 按钮。
        3.  系统将显示预测的类别。
        """)


if __name__ == "__main__":
    main()