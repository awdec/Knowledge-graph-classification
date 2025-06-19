# Knowledge-graph-classification

## 项目简介：

**数据获取与处理：**

​	使用爬虫（requests、bs4）爬取了历年 AAAI、CVPR、ICML、KDD、NeurlPS 若干。

​	使用 pdfminer，将爬取的 pdf 转换成 txt。然后用Re正则表达式对文本数据进行初步处理（数据清洗）。

​	使用 LangChain 的 RecursiveCharacterTextSplitter 对文本切分成多个片段（chunk），每片段长度约1500字符，重叠150字符。documents2Dataframe 函数将切分后的文本片段转为 DataFrame。

利用 LLM，调用 openAI api，使用 gpt-3.5-turbo模型抽取文本中概念及其关系，形成图的边列表。

​	用 NetworkX 创建无向图，加入节点和边，并对边赋予权重。用 NetworkX 的 Girvan-Newman 算法进行社区检测（划分社区）。给每个社区分配颜色和组号（用 seaborn 生成颜色调色板，并打乱分配）。

​	用 PyVis 把 NetworkX 图转成可交互的HTML图可视化，保存成文件。

​	最后将节点信息、边信息、社区信息保存到 CSV 文件，再处理整合成三元组，用于后续的图谱分类。

**图谱分类（graph-level）：**

**提取特征：**

​	基于 PyTorch Geometric 和 Sentence-BERT 的图神经网络文本分类模型，用于从结构化图（CSV边列表）中提取文本语义并训练 GINEConv 图神经网络进行分类。

​	对处理后的 graph.csv 文件进行分类，graph.csv 前三列分别是 node_1、node_2、edge。

​	对节点文本编码：x ← 使用 SentenceTransformer 编码为向量。对边属性编码：edge_attr ← 同样用 SentenceTransformer 对边文本编码。对边索引构造：edge_index ← 节点编号重映射后生成边。SentenceTransformer 使用 paraphrase-MiniLM-L6-v2。

​	数据划分为训练、验证、测试集。训练集占 70%，验证集占 15%，测试集占 15%，使用 DataLoader 批处理图结构数据。

​	构建 GINE 图神经网络，使用两层 GINEConv：节点特征和边特征均由 SentenceTransformer 生成。包含 edge_lin 先对边特征线性投影。node_mlp1/2用于 GINEConv 中的 MLP。global_mean_pool将图中所有节点聚合为图向量。self.lin 输出分类。

**训练过程：**

​	优化器：Adam（含 weight decay）。

​	损失函数：CrossEntropyLoss（用于多分类）。

​	早停优化：当验证集精度连续多轮不提升时停止训练（避免过拟合）。

**评估与可视化：**

​	指标：Accuracy、Precision、Recall、F1（macro）、ROC AUC（支持二分类和多分类）

​	图表绘制：Loss 曲线（随 epoch 变化）；ROC 曲线：二分类：标准 ROC 曲线；多分类：使用 One-vs-Rest 绘制每类 ROC 曲线

## 使用方法：

## 数据集：

kaggle：https://www.kaggle.com/datasets/largerice16pro/knowledge-graph

## 联系方式：

QQ：1120571672

邮箱：1120571672@qq.com
