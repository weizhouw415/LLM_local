from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import logging

MODEL_DIR = "/root/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda:6"

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局加载模型和分词器
logger.info("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype='auto',
    device_map=DEVICE
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
logger.info("Model and tokenizer loaded.")


@app.route('/api/run', methods=["POST"])
def run():
    start = time.time()
    try:
        data = request.get_json()
        logger.info("Received request: %s", data)
        
        # 检查输入参数
        if not data:
            logger.error("Parse input data failed.")
            return jsonify({'ret_code': -2, 'message': 'Parse input data failed.'}), 400
        for key in ["question", "prompt"]:
            if key not in data:
                logger.error("Missing required key: %s", key)
                return jsonify({'ret_code': -3, 'message': f"No required key: {key}"}), 400
        
        question = data['question']
        prompt = data["prompt"]
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        
        # 使用分词器的 apply_chat_template 方法将上面定义的消息列表转换为文本
        text = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 将处理后的文本令牌化并转换为模型输入张量，然后将这些张量移至指定设备
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        
        # 对输出进行解码
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = response[0].split("assistant\n\n")[1]
        end = time.time()
        t = str(round(end - start, 2)) + " s"
        resp = {"ret_code": 0, "message": response, "time": t}
        
        logger.info("Response: %s", resp)
        return jsonify(resp), 200
    except Exception as e:
        logger.exception("An error occurred")
        return jsonify({'ret_code': -1, 'message': str(e)}), 500


@app.route('/api/stream', methods=["POST"])
def stream_run():
    try:
        data = request.get_json()
        logger.info("Received request: %s", data)
        
        # 检查输入参数
        if not data:
            logger.error("Parse input data failed.")
            return jsonify({'ret_code': -2, 'message': 'Parse input data failed.'}), 400
        for key in ["question", "prompt"]:
            if key not in data:
                logger.error("Missing required key: %s", key)
                return jsonify({'ret_code': -3, 'message': f"No required key: {key}"}), 400
        
        question = data['question']
        prompt = data["prompt"]
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        
        def generate_responses():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            for i in range(generated_ids.size(1)):
                partial_ids = generated_ids[:, :i+1]
                partial_response = tokenizer.batch_decode(partial_ids, skip_special_tokens=True)[0]
                yield f"data: {partial_response}\\n\\n"
                time.sleep(0.1)
                logger.info("Partial response: %s", partial_response)
        
        return Response(generate_responses(), content_type='text/event-stream')
    except Exception as e:
        logger.exception("An error occurred")
        return jsonify({'ret_code': -1, 'message': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
