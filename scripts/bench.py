import time
import json
from contextlib import nullcontext
from pathlib import Path
import sys
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader

from model import GPTConfig, GPT

#============================================================================
dataset = 'sci-fi'
# 模型
n_layer = 12  # Transformer层数
n_head = 12  # 注意力头数
n_embd = 768  # 嵌入维度
dropout = 0.0  # dropout率，预训练时0较好，微调时可尝试0.1+
attention_dropout = 0.0
bias = False  # 是否在LayerNorm和Linear层中使用偏置
gradient_accumulation_steps = 5*8  # 用于模拟更大批次大小的梯度累积步数
batch_size = 6  # 如果gradient_accumulation_steps > 1，这是微批次大小
block_size = 1024  # 序列长度
RoPE=True #是否采用旋转编码,不然用可学习的位置编码
#-----------------------------------------------------------------------------
seed = 666
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
profile = True
#=============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
BASE_DIR = Path(__file__).resolve().parent


meta_path = BASE_DIR.parent / "data" / "processed" / f"{dataset}" / "meta.json"
with open(meta_path, 'r', encoding='utf-8') as file:
	mata_config = json.load(file)
vocab_size = mata_config['vocab_size']  # 词典大小
data_dir= BASE_DIR.parent / "data"/"processed"/f"{dataset}"/"train.npy" #数据路径

def get_batch():		#获取数据
	data = np.memmap(data_dir, dtype=np.int32, mode='r')
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
	y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
	if device_type == 'cuda':
		# pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
		x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
	else:
		x, y = x.to(device), y.to(device)
	return x, y
# train_dataset = MyDataset(data_dir, block_size=block_size)
# train_sampler = RandomSampler(train_dataset)
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=6,
#     sampler=train_sampler,
#     num_workers=0,
#     pin_memory=True,
#     drop_last=True,
# )
# train_iter = iter(train_loader)
# def get_batch():
# 	global train_iter, val_iter
# 	try:
# 		x, y = next(train_iter)  # 从训练迭代器取数据
# 	except StopIteration:
# 		train_iter = iter(train_loader)
# 		x, y = next(train_iter)
# 	# 转移到设备
# 	if device_type == 'cuda':
# 		x = x.pin_memory().to(device, non_blocking=True)
# 		y = y.pin_memory().to(device, non_blocking=True)
# 	else:
# 		x = x.to(device)
# 		y = y.to(device)
# 	return x, y

#init model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
				  bias=bias, vocab_size=vocab_size, dropout=dropout,attention_dropout=attention_dropout,RoPE=RoPE)

#模型初始化
gptconfig=GPTConfig(**model_args)
model = GPT(gptconfig)
model.to(device)
optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
	print("Compiling model...")
	model = torch.compile(model)

if profile:
	wait, warmup, active = 5, 5, 5
	num_steps = wait + warmup + active
	with torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
		on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
		record_shapes=False,
		profile_memory=False,
		with_stack=False, # incurs an additional overhead, disable if not needed
		with_flops=True,
		with_modules=False, # only for torchscript models atm
	) as prof:

		X, Y = get_batch()
		for k in range(num_steps):
			with ctx:
				logits, loss = model(X, Y)
			X, Y = get_batch()
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()
			lossf = loss.item()
			print(f"{k}/{num_steps} loss: {lossf:.4f}")

			prof.step()
else:

	# simple benchmarking
	torch.cuda.synchronize()
	for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
		t0 = time.time()
		X, Y = get_batch()
		for k in range(num_steps):
			with ctx:
				logits, loss = model(X, Y)
			X, Y = get_batch()
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()
			lossf = loss.item()
			print(f"{k}/{num_steps} loss: {lossf:.4f}")
		torch.cuda.synchronize()
		t1 = time.time()
		dt = t1-t0
		mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
		if stage == 1:
			print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")