import fasttext
import torch

state_dict = 'data/Comments/saved_dict/FastText.ckpt'
model_path = 'data/Comments/saved_dict/fasttext.bin'

# 加载 PyTorch 模型
model_state_dict = torch.load(state_dict)

# 创建 FastText 模型
model = fasttext.load_model('')

# 将 PyTorch 模型参数加载到 FastText 模型中
model.set_params(model_state_dict)

# 保存模型为 .bin 文件
model.save_model(model_path)
