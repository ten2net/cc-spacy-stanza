import stanza
import spacy_stanza
import pandas as pd
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from collections import Counter

import spacy

# 下载模型

# Download the stanza model if necessary
# stanza.download("zh-hans")

# 初始化管道
# nlp = spacy_stanza.load_pipeline("zh-hans")
nlp = spacy.load("zh_core_web_lg")

# 假设您的卷宗资料文本存储在一个名为documents的列表中
documents = ["七七事变，又称卢沟桥事变，发生于1937年7月7日 [1] 。1937年7月7日夜，卢沟桥的日本驻军在未通知中国地方当局的情况下，径自在中国驻军阵地附近举行所谓军事演习，并诡称有一名日军士兵失踪，要求进入北平西南的宛平县城搜查 [13] ，被中国驻军严词拒绝，日军随即向宛平城和卢沟桥发动进攻。中国驻军第29军37师219团奋起还击，进行了顽强的抵抗 [2] 。“七七事变”揭开了全国抗日战争的序幕", 
             "1932年1月28日午夜，日本海军第一遣外舰队司令盐泽幸一指挥海军陆战队分三路突袭上海闸北，第十九路军在总指挥蒋光鼐、军长蔡廷锴指挥下奋起抵抗，一·二八淞沪抗战爆发。",
             "1948年11月23日，东北野战军主力由锦州、营口、沈阳等地出发，隐蔽向北平、天津、唐山、塘沽地区开进。25日，华北军区第3兵团司令员杨成武、政治委员李井泉率第1、第2、第6纵队由集宁地区东进。29日，平津战役开始，华北军区第3兵团首先向张家口外围国民党军发起攻击，至12月2日，对张家口形成包围态势。"]

# 创建空的数据框
data = pd.DataFrame(columns=["卷宗","日期", "时间", "事件", "GPE","FAC","ORG"])

# 统计TOPK
def find_topN(doc,ent_type,k=1):
    c = Counter()
    for ent in doc.ents:
        if ent.label_ == ent_type:
            c[ent.text] += 1
    return c.most_common(k)

ent_types = ["DATE", "TIME", "EVENT","GPE", "FAC", "LOC","ORG", "PERSON"]
for i, doc_text in enumerate(documents):
    # 对每个卷宗文本进行命名实体识别
    doc = nlp(doc_text)   
    infomations = [find_topN(doc,ent_type,1) for ent_type in ent_types]    
    # infomations=[info[0][0] for info in infomations if len(info)>0]
    infomations=[info[0][0] if len(info)>0 else "" for info in infomations ]
    print(infomations)    
    data = data._append({
        "卷宗": f"卷宗{i+1}",
        "日期": infomations[0],
        "时间": infomations[1],
        "事件": infomations[2],
        "GPE": infomations[3],  
        "FAC": infomations[4],
        "ORG": infomations[5]
    }, ignore_index=True)
    
    
    
    # 提取命名实体、时间、类型、事件和位置信息，并添加到数据框中
    # for ent in doc.ents:
    #     data = data._append({
    #         "卷宗": f"卷宗{i+1}",
    #         "时间": ent.text,  # 根据实际情况提取时间信息
    #         "类型": ent.label_,
    #         "事件": "",  # 根据实际情况提取事件信息
    #         "位置": ent.start_char  # 根据实际情况提取位置信息
    #     }, ignore_index=True)
data.to_csv("命名实体信息2-2.csv", index=False) 

# from collections import Counter

# # 统计人物角色实体TOP10
# def find_role(doc):
#     c = Counter()
#     for ent in doc.ents:
#         if ent.label_ == 'PERSON':
#             c[ent.text] += 1
#     return c.most_common(10)

# print(find_role(doc))


# from spacy import displacy
# # 直接显示在Jupyter Notebook中
# displacy.render(doc, style='ent', jupyter=True) 


# # 定义读取文件操作
# def read_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()

# # 加载文本数据
# text = read_file("./data/Animal Farm.txt")  

# NER任务有哪些常见的应用场景？
# 命名实体识别（Named Entity Recognition，NER）在自然语言处理中有许多常见的应用场景。以下是一些常见的NER应用场景：
# 信息抽取（Information Extraction）：NER可以用于从文本中抽取特定类型的实体，如人名、地名、组织名等，以获取有关这些实体的结构化信息。
# 问答系统（Question Answering）：NER可以帮助识别问题中的命名实体，并根据这些实体提供准确的答案。
# 文本分类（Text Classification）：NER可以用于将文本分类到特定的实体类型，如将新闻文章分类为人名、组织名或地名等。
# 舆情分析（Sentiment Analysis）：NER可以用于识别评论或社交媒体文本中的人名、地名等实体，并将其与情感分析相结合，以了解人们对特定实体的情感倾向。
# 机器翻译（Machine Translation）：NER可以帮助在翻译过程中识别和保留命名实体的信息，以提高翻译质量和准确性。
# 自动摘要（Automatic Summarization）：NER可以帮助识别文本中的重要实体，从而生成更准确和有信息量的自动摘要。
# 搜索引擎优化（Search Engine Optimization）：NER可以用于识别网页内容中的命名实体，并提供更准确的搜索结果和关联性。
# 实体链接（Entity Linking）：NER可以将命名实体与知识图谱或数据库中的实体进行链接，以提供更丰富的上下文信息和知识关联。
# 这些只是一些常见的NER应用场景示例，实际上NER在许多领域和任务中都有广泛的应用，如金融、医疗、法律、新闻等。根据具体的应用需求，NER可以用于提取关键信息、实体关系分析、文本理解和增强搜索等方面。
# NER任务的评估指标有哪些？
# NER任务中常用的工具或库有哪些？
# 在命名实体识别（Named Entity Recognition，NER）任务中，常用的评估指标包括以下几种：
# 准确率（Precision）：准确率是指被正确标注的命名实体数量与系统标注的命名实体数量之比。准确率衡量了系统标注的命名实体中有多少是正确的。
# 召回率（Recall）：召回率是指被正确标注的命名实体数量与标准答案中的命名实体数量之比。召回率衡量了系统能够正确找到多少标准答案中的命名实体。
# F1值（F1-score）：F1值是准确率和召回率的调和平均值，计算公式为：2 * (准确率 * 召回率) / (准确率 + 召回率)。F1值综合考虑了准确率和召回率，是一个常用的综合评估指标。
# 精确度-召回率曲线（Precision-Recall Curve）：精确度-召回率曲线以不同的阈值为基础，绘制了系统在不同召回率下的准确率。通过分析精确度-召回率曲线，可以选择最优的阈值来平衡准确率和召回率。
# 实体级别评估指标：除了上述指标，还可以对每个命名实体类型进行独立的评估。例如，针对不同的命名实体类型，可以计算它们各自的准确率、召回率和F1值。
# 这些评估指标可以帮助评估NER系统的性能和效果。在评估过程中，通常会使用标注好的测试数据集，并与标准答案进行比较以计算指标。值得注意的是，NER任务中的评估指标可以根据具体的任务需求和数据集特点进行调整和扩展。

# 在命名实体识别（Named Entity Recognition，NER）任务中，有几个常用的工具和库可用于实现NER模型的开发和应用。以下是其中一些常见的工具和库：
# SpaCy：SpaCy是一个流行的Python库，提供了高性能的自然语言处理功能，包括命名实体识别。它提供了预训练的NER模型，并且还允许用户进行自定义训练以适应特定的任务和领域。
# NLTK：Natural Language Toolkit（NLTK）是一个广泛使用的Python库，提供了丰富的自然语言处理工具和资源。NLTK包括许多NER相关的功能和数据集，可以用于开发和评估NER模型。
# Stanford NER：Stanford NER是斯坦福大学开发的一种NER工具，它提供了预训练的NER模型，可以识别多种类型的命名实体。Stanford NER还提供了Java和Python的接口，方便集成到自定义应用中。
# Hugging Face Transformers：Hugging Face Transformers是一个流行的自然语言处理库，提供了各种预训练的语言模型和NER模型。通过使用Hugging Face Transformers，可以轻松地加载和使用预训练的NER模型，如BERT、GPT等。
# AllenNLP：AllenNLP是一个用于自然语言处理研究和开发的开源库，它提供了丰富的工具和模型。AllenNLP包括NER模型的实现和预训练模型，同时还提供了可自定义的组件和训练框架。
# 这些工具和库提供了方便且强大的功能，能够加速NER模型的开发和应用。您可以根据自己的需求和偏好选择合适的工具，并参考它们的文档和示例代码来了解更多细节和用法。



# 命名实体识别（Named Entity Recognition，NER）任务的实体类型可以根据不同的语料库和任务需求而有所不同。下面是一些常见的命名实体类型：
# 人名（Person）：指具体的个人姓名，如"John Smith"。
# 地名（Location）：指特定的地理位置，如"New York"。
# 组织名（Organization）：指具体的组织、机构或公司名称，如"Microsoft"。
# 日期/时间（Date/Time）：指特定的日期或时间，如"2023-10-09"或"9:00 AM"。
# 货币（Money）：指特定的货币单位或金额，如"$100"。
# 百分比（Percent）：指特定的百分数，如"20%"。
# 数量（Quantity）：指特定的数量或计量单位，如"5 kilograms"。
# 产品（Product）：指特定的产品、商品或物品名称，如"iPhone"。
# 事件（Event）：指特定的事件、活动或竞赛名称，如"Olympic Games"。
# 专业术语（Miscellaneous）：指特定的领域术语、技术术语或其他类别中的实体，如"DNA"。
# 请注意，这只是一些常见的命名实体类型示例，实际应用中可能会根据任务和领域的不同而有所变化。在使用特定的命名实体识别工具或库时，可以查看其文档以了解支持的实体类型列表。    


# 大纲：
# 一、实体类型
# 1. PER(人物, Person)
# 2. ORG(组织机构, Organization)
# 3. C(地理/社会/政治实体,Geo-Political)
# 4. LOC(处所,Locations)
# 5. FAC(设施,Facilities)
# 6. VEH(交通工具,Vehicle)
# 7. WEA(武器,Weapon)
# 二、关系类型
# 1. Physical (物理位置关系)
# 2. Part-whole(部分-整体关系)
# 3. Personal-Social（ 人物-社会关系）
# 4. ORG-Affiliation
# 5. Agent-Artifact(施事关系)
# 6. General-Affiliation( 通用附属关系)


# https://zhuanlan.zhihu.com/p/354779308

# nlp = spacy.load("zh_core_web_sm")
# user_dict = get_user_dict('c:/user_dict.txt')
# nlp.tokenizer.pkuseg_update_user_dict(user_dict)
# ruler = nlp.add_pipe('entity_ruler')
# patterns = [{"label": "A", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "电流"}]},\
#             {"label": "T", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "温度"}]}]
# ruler.add_patterns(patterns)
# doc = nlp('启动一台真空泵，检查真空泵电机电流、入口压力、轴承温度、盘根温度、电机线圈温度、声音、振动正常。')
# print([(ent.text, ent.label_) for ent in doc.ents])

# #其结果为：[('真空泵电机电流', 'A'), ('轴承温度', 'T'), ('盘根温度', 'T'), ('电机线圈温度', 'T')]