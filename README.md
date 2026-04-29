# Kaggle Word2Vec NLP Tutorial Competition

## 比赛要求

这是Kaggle上的"Bag of Words Meets Bags of Popcorn"情感分析比赛，要求参与者开发模型来预测电影评论的情感（正面或负面）。

### 任务
- 基于电影评论文本，预测其情感倾向（0=负面，1=正面）
- 评估指标：AUC-ROC分数

### 数据格式
- 训练数据：labeledTrainData.tsv（包含id、情感标签和评论文本）
- 测试数据：testData.tsv（包含id和评论文本，需要预测情感标签）

## 代码实现

### 方法
本代码实现了以下流程：
1. **数据清洗**：移除HTML标签、URL、标点和数字，保留否定词
2. **分词**：使用短语模式（如"not_good"、"very_good"等）
3. **Word2Vec**：训练词向量模型
4. **均值Embedding**：将评论中的词向量求平均值作为特征
5. **逻辑回归**：使用简单的逻辑回归模型进行分类

### 依赖库
- pandas
- numpy
- bs4 (BeautifulSoup)
- gensim (Word2Vec)
- scikit-learn

### 如何运行

1. **下载数据集**：
   - 访问 Kaggle 比赛页面：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data
   - 下载以下文件并解压到代码所在目录：
     - labeledTrainData.tsv
     - testData.tsv

2. **安装依赖**：
   ```
   pip install pandas numpy beautifulsoup4 gensim scikit-learn
   ```

3. **运行代码**：
   ```
   python Word2Vec.py
   ```

4. **查看结果**：
   - 代码会生成 `submission.csv` 文件，包含测试数据的预测结果
   - 可以将此文件上传到 Kaggle 比赛页面进行评估

### 代码特点

1. **保留否定词**：在预处理过程中，特别保留了"not"等否定词，以捕捉情感的反转
2. **短语模式**：在分词时使用了短语模式，将常见的双词短语（如"not good"）组合为单个 token
3. **简单模型**：使用逻辑回归作为分类器，确保模型简单且易于理解
4. **鲁棒性**：不依赖NLTK资源下载，使用手动定义的停用词列表

### 模型性能

代码会在训练过程中计算验证集的准确率和AUC分数，以评估模型性能。最终的预测结果会保存到 `submission.csv` 文件中，可直接上传到Kaggle比赛页面。

## 注意事项

- 首次运行时，Word2Vec模型训练可能需要一些时间
- 确保数据集文件已正确下载并放置在代码所在目录
- 代码使用了默认参数，可根据需要调整Word2Vec和逻辑回归的参数以获得更好的性能