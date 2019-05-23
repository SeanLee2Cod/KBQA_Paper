# KBQA_Paper
---
the paper trace of KBQA, also contains some common/classic NLP/Deep Learning papers and resources.

## 一 传统的方法
---
### 1. 语义解析（Semantic Parsing）

该方法是一种偏linguistic的方法，主体思想是将自然语言转化为一系列形式化的逻辑形式（logic form）,通过对逻辑形式进行自底向上的解析，得到一种可以表达整个问题语义的逻辑形式，通过相应的查询语句（类似lambda-Caculus）在知识库中进行查询，从而得出答案。

> * Berant J, Chou A, Frostig R, et al. Semantic Parsing on Freebase from Question-Answer Pairs[C]//EMNLP. 2013, 2(5): 6.

> * Cai Q, Yates A. Large-scale Semantic Parsing via Schema Matching and Lexicon Extension[C]//ACL (1). 2013: 423-433.

> * Kwiatkowski T, Choi E, Artzi Y, et al. Scaling semantic parsers with on-the-fly ontology matching[C]//In Proceedings of EMNLP. Percy. 2013.

> * Fader A, Zettlemoyer L, Etzioni O. Open question answering over curated and extracted knowledge bases[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 1156-1165.

### 2. 信息抽取（Information Extraction）

该类方法通过提取问题中的实体，通过在知识库中查询该实体可以得到以该实体节点为中心的知识库子图，子图中的每一个节点或边都可以作为候选答案，通过观察问题依据某些规则或模板进行信息抽取，得到问题特征向量，建立分类器通过输入问题特征向量对候选答案进行筛选，从而得出最终答案。

> * Yao X, Van Durme B. Information Extraction over Structured Data: Question Answering with Freebase[C]//ACL (1). 2014: 956-966.

### 3. 向量建模（Vector Modeling）

该方法思想和信息抽取的思想比较接近，根据问题得出候选答案，把问题和候选答案都映射为**分布式表达（Distributed Embedding）**，通过训练数据对该分布式表达进行训练，使得问题和正确答案的向量表达的得分（通常以点乘为形式）尽量高,模型训练完成后则可根据候选答案的向量表达和问题表达的得分进行筛选，得出最终答案。

> * Antoine Bordes, Sumit Chopra, Jason Weston:
Question Answering with Subgraph Embeddings. EMNLP 2014: 615-620


> * Yang M C, Duan N, Zhou M, et al. Joint Relational Embeddings for Knowledge-based Question Answering[C]//EMNLP. 2014, 14: 645-650.

> * Bordes A, Weston J, Usunier N. Open question answering with weakly supervised embedding models[C]//Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer Berlin Heidelberg, 2014: 165-180.

## 二 基于深度学习的KBQA方法
---
### 1. 使用CNN对语义解析方法提升

> * Yih S W, Chang M W, He X, et al. Semantic parsing via staged query graph generation: Question answering with knowledge base[J]. 2015. 
(注：该paper来自微软，是ACL 2015年的Outstanding paper，也是目前（）KB-QA效果最好的paper之一)

### 2. 使用CNN对向量建模方法进行提升

> * Dong L, Wei F, Zhou M, et al. Question Answering over Freebase with Multi-Column Convolutional Neural Networks[C]//ACL (1). 2015: 260-269.

### 3. 使用LSTM、CNN进行实体关系分类

> * Xu Y, Mou L, Li G, et al. Classifying Relations via Long Short Term Memory Networks along Shortest Dependency Paths[C]//EMNLP. 2015: 1785-1794.

> * Zeng D, Liu K, Lai S, et al. Relation Classification via Convolutional Deep Neural Network[C]//COLING. 2014: 2335-2344.（Best paper）

> * Zeng D, Liu K, Chen Y, et al. Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks[C]//EMNLP. 2015: 1753-1762.

### 4. 使用记忆网络（Memory NetWorks），注意力机制（Attention Mechanism）进行KBQA

> * Bordes A, Usunier N, Chopra S, et al. Large-scale simple question answering with memory networks[J]. arXiv preprint arXiv:1506.02075, 2015.

> * Zhang Y, Liu K, He S, et al. Question Answering over Knowledge Base with Neural Attention Combining Global Knowledge Information[J]. arXiv preprint arXiv:1606.00979, 2016.

（注：以上论文皆可通过[DBLP](https://dblp.uni-trier.de/)搜索并下载获取全文pdf）

## 三 核心概念与实用工具
---

### 1. 核心概念

#### （1）NLP基础

> * [组合范畴语法（Combinatory Categorical Grammars，CCG）](https://zh.wikipedia.org/wiki/%E7%BB%84%E5%90%88%E8%8C%83%E7%95%B4%E8%AF%AD%E6%B3%95)

> * [数据归一化：SUTime](https://link.zhihu.com/?target=http%3A//nlp.stanford.edu/pubs/lrec2012-sutime.pdf)

> * [语法依存树（Dependency tree）](https://nlpcs.com/article/syntactic-parsing-by-dependency)

> * [词袋模型（Bag-of-words model）](https://blog.csdn.net/v_JULY_v/article/details/6555899)




#### （2）深度学习基础

> * [AdaGrad算法](https://zhuanlan.zhihu.com/p/29920135)

> * [lambda-rank算法](https://link.zhihu.com/?target=https%3A//pdfs.semanticscholar.org/0df9/c70875783a73ce1e933079f328e8cf5e9ea2.pdf)

#### （3）Deep Learning for NLP 

> * [词嵌入向量（word-embedding）](https://link.zhihu.com/?target=http%3A//papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

> * [text-CNNs](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1408.5882)

> * [character-CNNs](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1509.01626.pdf)

> * [神经图灵机（Neural Tuning Machine）](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1410.5401)

> * [端到端学习（End-to-End）的记忆网络](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1503.08895.pdf)

> * [注意力机制应用于NLP问题：提出经典的encoder-decoder with attention mechanism模型](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1606.00979)

> * [知识图谱补全的经典方法:TransE](https://link.zhihu.com/?target=https%3A//www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)


### 2. 实用工具

> * [Stanford CoreNLP – Natural language software](https://stanfordnlp.github.io/CoreNLP/)



## 四 推荐阅读与学习资源
---
### 1. 推荐阅读

> * [肖仰华 | 基于知识图谱的问答系统](https://blog.csdn.net/TgqDT3gGaMdkHasLZv/article/details/78146295?%3E)

> * [技术动态 | 基于深度学习知识库问答研究进展](http://blog.openkg.cn/%E6%8A%80%E6%9C%AF%E5%8A%A8%E6%80%81-%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9F%A5%E8%AF%86%E5%BA%93%E9%97%AE%E7%AD%94%E7%A0%94%E7%A9%B6%E8%BF%9B%E5%B1%95/#more-394)

> * [TF-IDF与余弦相似性的应用（一）：自动提取关键词](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html)

> * [揭开知识库问答KB-QA的面纱8·非结构化知识篇](https://zhuanlan.zhihu.com/p/26650719)

> * [揭开知识库问答KB-QA的面纱9·动态模型篇](https://zhuanlan.zhihu.com/p/27105336)

### 2. 学习资源

> * [Your new Mentor for Data Science E-Learning](https://github.com/virgili0/Virgilio)

> * [增强学习](https://zhuanlan.zhihu.com/intelligentunit)

## 五 参考链接
---

> * [知乎专栏：揭开知识库问答KB-QA的面纱](https://zhuanlan.zhihu.com/kb-qa)

