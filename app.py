from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_DIR = "/root/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda:6"

app = Flask(__name__)

# model dir 是模型文件所在的目录。# torch_dtype="auto" 自动选择最优的数据类型以平衡性能和精度。# device_map="auto" 自动将模型的不同部分映射到可用的设备上。
model= AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype='auto',
    device_map=DEVICE
)

# 加载与模型相匹配的分词器。分词器用于将文本转换成模型能够理解和处
tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR)

@app.route('/api/run', methods = ["POST"])
def run():
    try:
        start = time.time()
        data = request.get_json()
        # 检查输入参数
        if not data:
            return jsonify({'ret_code': -2, 'message': 'Parse input data failed.'}), 400
        for key in ["question", "prompt"]:
            if key not in data:
                return jsonify({'ret_code': -3, 'message': f"No required key: {key}"}), 400
        quesiton = data['question']
        prompt = data["prompt"]
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": quesiton}
        ]
        # 使用分词器的 apply_chat_template 方法将上面定义的消息列表转护# tokenize=False 表示此时不进行令牌化，add_generation_promp
        text =tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True
        )
        
        #将处理后的文本令牌化并转换为模型输入张量，然后将这些张量移至之前
        model_inputs=tokenizer([text],return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        # 对输出进行解码
        response=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        end = time.time()
        t = str(round(end - start, 2)) + " s"
        resp = {"ret_code": 0, "message": response, "time": t}
        return jsonify(resp), 200
    except Exception as e:
        return e
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)