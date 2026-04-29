import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
import os

# 手动定义停用词列表
ENGLISH_STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
])

# 否定词列表
NEGATIONS = {"not", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "no", "never", "nor"}

# 从停用词中移除否定词
STOP_WORDS = ENGLISH_STOPWORDS - NEGATIONS

# 情感词典
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'fantastic', 'amazing', 'wonderful', 'awesome', 'brilliant',
    'perfect', 'outstanding', 'superb', 'terrific', 'fabulous', 'lovely', 'enjoyable', 'delightful',
    'positive', 'happy', 'glad', 'pleased', 'satisfied', 'content', 'joyful', 'excited',
    'thrilled', 'impressed', 'moved', 'touched', 'incredible', 'unbelievable', 'remarkable', 'exceptional'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'horrendous', 'dreadful', 'disgusting', 'pathetic',
    'horrid', 'abysmal', 'atrocious', 'appalling', 'lousy', 'poor', 'inferior', 'subpar',
    'negative', 'sad', 'unhappy', 'upset', 'disappointed', 'dissatisfied', 'frustrated', 'angry',
    'mad', 'annoyed', 'irritated', 'bored', 'tired', 'weary', 'exhausted', 'depressed',
    'gloomy', 'miserable', 'hopeless', 'worthless', 'pointless', 'meaningless', 'useless'
}

# 简单的分词函数
def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# 简单的词形还原函数
def simple_lemmatize(word):
    lemmatization_rules = {
        'ing$': '',
        'ed$': '',
        'es$': '',
        's$': ''
    }
    
    for suffix, replacement in lemmatization_rules.items():
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement
    
    return word

# 读取数据
def load_data():
    train_path = "labeledTrainData.tsv"
    test_path = "testData.tsv"
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please download the dataset from Kaggle.")
        print("Dataset URL: https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data")
        exit(1)
    
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found. Please download the dataset from Kaggle.")
        print("Dataset URL: https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data")
        exit(1)
    
    train = pd.read_csv(train_path, delimiter="\t", quoting=3)
    test = pd.read_csv(test_path, delimiter="\t", quoting=3)
    
    return train, test

# 数据清洗函数，根据预处理注意事项进行优化
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # 1. 移除HTML标签（注意事项1）
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # 2. 移除URL和邮箱
    text = re.sub(r'http\S+|www\.\S+|[\w\.-]+@[\w\.-]+', '', text)
    
    # 3. 保留情感表达的标点，处理否定形式（注意事项3）
    # 首先将否定词标记出来
    text = re.sub(r"(\bnot\b|\bdon't\b|\bdoesn't\b|\bdidn't\b|\bwon't\b|\bwouldn't\b|\bcan't\b|\bcannot\b|\bshouldn't\b|\bcouldn't\b|\bisn't\b|\baren't\b|\bwasn't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bhadn't\b|\bno\b|\bnever\b|\bnor\b)", r"NEG_\1", text)
    
    # 4. 移除数字
    text = re.sub(r"\d+", ' ', text)
    
    # 5. 转换为小写（注意事项2）
    text = text.lower()
    
    # 6. 恢复否定词
    text = re.sub(r"NEG_(\w+)", r"\1", text)
    
    # 7. 移除多余的空格
    text = re.sub(r"\s+", ' ', text).strip()
    
    return text

# 分词函数，使用短语模式
def tokenize_with_phrases(text):
    # 1. 扩展的短语检测：将更多情感相关的短语组合
    phrases = [
        "not good", "not bad", "very good", "very bad", "really good", "really bad", 
        "never good", "never bad", "extremely good", "extremely bad", "quite good", "quite bad",
        "so good", "so bad", "too good", "too bad", "pretty good", "pretty bad",
        "fairly good", "fairly bad", "rather good", "rather bad", "reasonably good", "reasonably bad",
        "incredibly good", "incredibly bad", "amazingly good", "amazingly bad", "exceptionally good", "exceptionally bad"
    ]
    
    for phrase in phrases:
        if phrase in text:
            text = text.replace(phrase, phrase.replace(' ', '_'))
    
    # 2. 分词
    tokens = simple_tokenize(text)
    
    # 3. 词形还原（注意事项6）
    tokens = [simple_lemmatize(token) for token in tokens]
    
    # 4. 移除停用词，但保留否定词（注意事项4, 5）
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    
    return tokens

# 提取情感特征
def extract_sentiment_features(tokens):
    # 计算正面词和负面词的数量
    positive_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    
    # 计算否定词的数量
    negation_count = sum(1 for token in tokens if any(neg in token for neg in NEGATIONS))
    
    # 计算情感得分
    sentiment_score = positive_count - negative_count
    
    # 归一化
    total_words = len(tokens)
    if total_words > 0:
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        negation_ratio = negation_count / total_words
    else:
        positive_ratio = 0
        negative_ratio = 0
        negation_ratio = 0
    
    return np.array([positive_count, negative_count, negation_count, sentiment_score, positive_ratio, negative_ratio, negation_ratio])

# 训练Word2Vec模型
def train_word2vec(tokens_list):
    model = Word2Vec(
        sentences=tokens_list,
        vector_size=300,  # 增加向量维度
        window=10,        # 增加窗口大小
        min_count=2,       # 减少最小词频
        epochs=30,         # 增加训练轮数
        workers=4
    )
    return model

# 生成词向量的平均值
def get_mean_embedding(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if not vectors:
        return np.zeros(300)
    
    return np.mean(vectors, axis=0)

# 主函数
def main():
    # 加载数据
    print("Loading data...")
    train, test = load_data()
    
    # 数据清洗
    print("Cleaning text...")
    train['clean_review'] = train['review'].apply(clean_text)
    test['clean_review'] = test['review'].apply(clean_text)
    
    # 分词
    print("Tokenizing text...")
    train['tokens'] = train['clean_review'].apply(tokenize_with_phrases)
    test['tokens'] = test['clean_review'].apply(tokenize_with_phrases)
    
    # 训练Word2Vec模型
    print("Training Word2Vec model...")
    tokens_list = train['tokens'].tolist() + test['tokens'].tolist()
    w2v_model = train_word2vec(tokens_list)
    
    # 生成特征向量
    print("Generating feature vectors...")
    
    # 生成Word2Vec特征
    X_train_w2v = np.array([get_mean_embedding(tokens, w2v_model) for tokens in train['tokens']])
    X_test_w2v = np.array([get_mean_embedding(tokens, w2v_model) for tokens in test['tokens']])
    
    # 生成情感特征
    X_train_sentiment = np.array([extract_sentiment_features(tokens) for tokens in train['tokens']])
    X_test_sentiment = np.array([extract_sentiment_features(tokens) for tokens in test['tokens']])
    
    # 组合特征
    X_train = np.hstack([X_train_w2v, X_train_sentiment])
    X_test = np.hstack([X_test_w2v, X_test_sentiment])
    y_train = train['sentiment'].values
    
    # 分割验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 训练逻辑回归模型，调整参数
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        C=0.1,  # 调整正则化参数
        penalty='l2',  # L2正则化
        solver='liblinear'  # 适合小数据集的求解器
    )
    lr_model.fit(X_train_split, y_train_split)
    
    # 评估模型
    y_val_pred = lr_model.predict(X_val)
    y_val_pred_proba = lr_model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    
    # 预测测试数据
    print("Predicting test data...")
    test_pred = lr_model.predict(X_test)
    
    # 创建提交文件
    submission = pd.DataFrame({
        "id": test['id'],
        "sentiment": test_pred
    })
    
    # 保存结果
    output_path = "submission.csv"
    submission.to_csv(output_path, index=False, quoting=3)
    print(f"Submission file saved to: {output_path}")

if __name__ == "__main__":
    main()