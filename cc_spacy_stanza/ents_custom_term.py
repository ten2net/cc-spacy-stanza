import spacy
from spacy.tokens import Doc, Span

# 加载预训练的SpaCy模型
nlp = spacy.load('en_core_web_sm')

# 创建一个新的实体类型
new_entity_type = 'YOUR_ENTITY_TYPE'

# 添加特定领域术语到词汇表
terms = ['term1', 'term2', 'term3']
for term in terms:
    lexeme = nlp.vocab[term]
    lexeme.is_stop = False  # 可选：如果术语是停用词，则设置为False

# 定义自定义的实体规则函数
def custom_entity_rules(doc):
    # 遍历每个词符
    for i in range(len(doc)):
        # 检查词符是否匹配特定领域术语
        if doc[i].text in terms:
            # 获取匹配的术语的起始和结束索引
            start = i
            end = i + 1
            # 创建一个新的命名实体
            entity = Span(doc, start, end, label=new_entity_type)
            # 将命名实体添加到文档的实体列表中
            doc.ents = list(doc.ents) + [entity]
    return doc

# 添加自定义实体规则到SpaCy的处理管道
nlp.add_pipe(custom_entity_rules, after='ner')

# 示例文本
text = "This is a sample sentence containing term1 and term2."

# 处理文本
doc = nlp(text)

# 打印识别的命名实体
for ent in doc.ents:
    if ent.label_ == new_entity_type:
        print(ent.text, ent.label_)