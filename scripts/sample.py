#python scripts/sample.py --model nanoGPT-2 --config default.py
#=============================================================================
import argparse
import pickle
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys
import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from model import GPTConfig, GPT
from utils.tokenizer_utils import SimpleVocab
#==============================================================================
#命令行输入处理
parser = argparse.ArgumentParser(description="生成文本")
parser.add_argument("--model", type=str, required=True,
					help="对应experiment/下对应的文件夹")
parser.add_argument("--config", type=str,required=True,
					help="配置文件的名称:对应inference/config/下的python配置文件")
args = parser.parse_args()
#------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
model_dir=BASE_DIR.parent /"experiments"/args.model
config_path=BASE_DIR.parent /"inference"/ "configs" / args.config
# -----------------------------------------------------------------------------
name=f"{args.model}"+datetime.now().strftime("%Y%m%d-%H%M")
start = "\n" # 生成文本的起始内容
num_samples = 10 # 要生成的样本数量
max_new_tokens = 500 # 生成的最多的token数
temperature = 0.8 # 小于1则更保守，大于1则更激进
top_k = 200 # 生成文本时，只保留概率最高的前 k 个 token
seed = 1337 #随机数种子
device = 'cuda' # 设备，例如：'cpu', 'cuda', 'cuda:0', 'cuda:1'
compile = True # 是否编译
# -----------------------------------------------------------------------------
#读取配置
exec(open(config_path).read())
#------------------------------------------------------------------------------
timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type,dtype=ptdtype)
#------------------------------------------------------------------------------
#模型初始化
ckpt_path = model_dir / 'checkpoint.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint_model_args)
model = GPT(gptconf)
# 处理参数名前缀问题（DDP训练时可能出现）
state_dict = checkpoint['model']  # 从检查点中取出模型权重参数
unwanted_prefix = '_orig_mod.'  # DDP包装时可能自动添加的前缀
for k, v in list(state_dict.items()):
	if k.startswith(unwanted_prefix):
		# 移除前缀，确保参数名与当前模型匹配
		state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
# 加载处理后的权重到模型
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']
model.eval()
model.to(device) #转到GPU
if compile:
	print("compiling the model... (takes a ~minute)")
	model = torch.compile(model) # 编译模型
#--------------------------------------------------------------------------------
#加载词表
vocab_dir = BASE_DIR.parent / "data" / "vocab" / f"{checkpoint['vocab_name']}"
with open(vocab_dir, 'rb') as f:
	vocab = pickle.load(f)
#--------------------------------------------------------------------------------
output_file = BASE_DIR.parent/ "inference" / f"{name}.md"
config_info = {
    "模型名称": args.model,
    "配置文件": args.config,
    "生成时间": timestamp,
    "样本数量": num_samples,
    "最大生成长度": max_new_tokens,
    "温度参数": temperature,
    "top_k参数": top_k,
    "随机种子": seed,
    "使用设备": device,
    "模型迭代次数": iter_num,
    "最佳验证损失": float(best_val_loss),
    "词表名称": checkpoint["vocab_name"]
}
#--------------------------------------------------------------------------------
start_tokens=list(start)
start_ids = vocab.encode(start_tokens)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
# run generation
# 打开.md文件准备写入
with open(output_file, "w", encoding="utf-8") as f, torch.no_grad(), ctx:
	# 第一部分：配置信息（Markdown标题+列表）
	f.write("## 配置信息\n")
	f.write("| 参数         | 数值                 |\n")
	f.write("|--------------|----------------------|\n")
	for key, value in config_info.items():
		f.write(f"| {key} | {value} |\n")
	f.write("\n---\n\n")  # 分隔线
	f.write("## 提示词\n")
	f.write(f"{start}\n")
	# 第二部分：生成的文本结果（按样本编号区分）
	f.write("## 生成文本\n")
	for k in range(num_samples):
		y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
		text = "".join(vocab.decode(y[0].tolist()))
		print(f"样本 {k + 1}/{num_samples}")
		print(text)
		print('---------------')

		# 写入.md文件（样本编号+文本内容，用分隔线区分）
		f.write(f"## 样本 {k + 1}\n")
		f.write(text + "\n\n")
		f.write("---\n\n")  # 样本间分隔线