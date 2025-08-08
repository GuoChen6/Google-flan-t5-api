from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 模型文件夹路径
MODEL_PATH = "/home/wangru/MichealChen/test/googleflan-t5-small"

# 加载分词器和模型
print("开始加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
print("模型加载完成")

# 创建 FastAPI 应用
app = FastAPI()

# 请求体格式
class Query(BaseModel):
    text: str

# POST 接口
@app.post("/chat")
def chat(query: Query):
    print(f"收到请求：{query.text}")
    inputs = tokenizer(query.text, return_tensors="pt")
    print("分词完成，开始生成...")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("生成完成，开始解码...")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"返回结果：{response}")
    return {"response": response}

