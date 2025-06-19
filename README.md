# Knowledge-graph-classification

## 项目简介：

**数据获取与处理：**

​	使用爬虫（requests、bs4）爬取了历年 AAAI、CVPR、ICML、KDD、NeurlPS 若干。

​	使用 pdfminer，将爬取的 pdf 转换成 txt。然后用Re正则表达式对文本数据进行初步处理（数据清洗）。

​	使用 LangChain 的 RecursiveCharacterTextSplitter 对文本切分成多个片段（chunk），每片段长度约1500字符，重叠150字符。documents2Dataframe 函数将切分后的文本片段转为 DataFrame。

​	利用 LLM，调用 openAI api，使用 gpt-3.5-turbo 模型抽取文本中概念及其关系，形成图的边列表。

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

**爬虫：**

​	/spider/ 中存放 AAAI.py CVPR.py ICML.py KDD.py NeulPS.py 爬虫脚本，用于爬取对应文献。每个文献爬取 200 份或达到总量后结束。

​	AAAI 脚本重新启动需从头开始爬取，其余脚本可跳过已有内容。

​	爬取的文件以 pdf 格式，默认保存至 /spider/ 对应文件夹中，例如 AAAI 保存至 /spider/AAAI。

**数据清洗：**

​	PDFtoTXT.py 将 pdf 转换成 txt，自设置存储路径，目标文件夹需存在。

​	clean.py 先对 txt 进行编号（若已经编号可以手动跳过），后对生成的 txt 文件使用正则表达式进行处理，默认保存至原路径，可自行修改。

**创建图谱：**

​	create.py 调用 openAI api，使用 gpt-3.5-turbo（或自行选择其它模型） 模型抽取文本中概念及其关系，形成图的边列表。

​	在 /helpers/LLM 中设置你的 openAI api key（笔者使用 closeAI 三方代理），也可以使用 deepseek-chat（同样是 openAI api，修改 key 即可）。

​	可使用 test.py 测试你的 openAi api key 是否有效。

​	生成的 knowledge graph csv 文件存储在 data_output 中（用于后续图谱分类），更详细的 edge、node、communities 信息存在在 /node 文件夹中，而图谱图像文件以 html 格式存储在 /docs 文件夹中。

**分类前的再次数据清洗：**

​	graphtotuple.py 用于将 graph.csv 文件以 | 分割成 node_1、node_2、edge 三列形成三元组文件，进行后续的图谱分类任务。

**图谱分类：**

​	classification.py 构建 GINE 图神经网络对 graph.csv 进行图谱分类。

**其它：**

​	项目中还提供了 name.py 用于处理由于创建图谱的不统一，而在分类时统一 graph.csv 文件名。

​	以及 make.py 用于统一可能存在在文件编码问题（统一转换成 utf-8）。

​	style.py 对文本进行风格分析。

​	

​	

## 数据集：

kaggle：https://www.kaggle.com/datasets/largerice16pro/knowledge-graph

## 联系方式：

QQ：1120571672

邮箱：1120571672@qq.com
