from modelscope import snapshot_download
 
 # 下载模型参数
model_dir=snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct')
print(model_dir)
