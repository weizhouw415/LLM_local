from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/root/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda:6"

class Local:
    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def run(self, question: str, prompt: str = "") -> str:
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        try:
            # 使用分词器的 apply_chat_template 方法将上面定义的消息列表转换为文本
            text = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 将处理后的文本令牌化并转换为模型输入张量，然后将这些张量移至指定设备
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            
            # 对输出进行解码
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return response[0].split("assistant\n\n")[1]
        except Exception as e:
            print(f"Error: {e}")
            return ""
        

if __name__ == "__main__":
    llm = Local(MODEL_DIR, DEVICE)
    answer = llm.run("你好你是谁？", "用简体中文回答")
    print(answer)