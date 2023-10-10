# 实验二
import spacy
from spacy import displacy
from collections import Counter

# 加载中文语言模型 zh_core_web_trf
nlp = spacy.load("zh_core_web_trf")

# 定义读取文件操作
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 加载文本数据
text = read_file("./data/Animal Farm.txt")

# 将文本对象传给语言模型处理
doc = nlp(text)

# 将实体识别注释结果直接显示在Jupyter Notebook中
displacy.render(doc, style='ent', jupyter=True)

# 统计人物角色实体TOP10
def find_role(doc):
    c = Counter()
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            c[ent.text] += 1
    return c.most_common(10)

# 输出统计结果
print(find_role(doc))