# ddp启动方式示例
# # 单节点4卡
# torchrun --standalone --nproc_per_node=4 train.py
# torchrun --standalone --nproc_per_node=4 scripts/train.py --config default.py --init_from scratch

# # 2节点（例如节点0 IP为123.456.123.456）
# # 节点0（主节点）
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py --config default.py
# # 节点1（从节点）
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py --config default.py

#==========================================================================
import sys
from datetime import datetime
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader, Dataset
import json
import os
import time
import math
import pickle
from contextlib import nullcontext
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
#=============================================================================
#命令行输入处理
parser = argparse.ArgumentParser(description="模型训练")
parser.add_argument("--config", type=str,required=True,
					help="配置文件的名称:对应train/config/下的python配置文件")
parser.add_argument("--init_from", type=str, default="scratch",
						choices=["scratch", "resume"],
						help="初始化模式：scratch=从头训练，resume=从检查点恢复")	#由于open-ai GPT-2的权重不支持中文，故没有实现直接进行微调的功能
args = parser.parse_args()

#=============================================================================
BASE_DIR = Path(__file__).resolve().parent
data_dir=BASE_DIR.parent / "data"/"processed" #数据路径
config_path=BASE_DIR.parent /"train"/ "configs" / args.config
#-----------------------------------------------------------------------------
#实验基本情况
name="nano-GPT" #实验的名称：对应 experiment/ 下的实验数据保存的位置
dataset = "sci-fi"  #数据集名称：对应 processed/ 下的子目录（如 wiki_zh.plt、shakespeare）
#wandb
wandb_log=True
wandb_project = 'nano-GPT'
wandb_run_name = "run"
#-----------------------------------------------------------------------------
#读取数据方式
use_dataloader=False #是否用torch的dataloader来读取数据
#如果使用dataloader
if use_dataloader:
	num_workers=4  # 多进程加载（根据CPU核心数调整，建议4-8）
	pin_memory=True  # 锁页内存（加速GPU传输）
	drop_last=True  # 训练时丢弃最后一个不完整批次
	prefetch_factor=2  # 预取2个批次（提前加载数据）
#-----------------------------------------------------------------------------
#输出与评估
log_interval=1 #输出loss的间隔
eval_interval=10 #经过多少次进行一次评估
eval_iters=20 #每一次评估迭代多少次
always_save_checkpoint=False #是否一直保存breakpoint

#-----------------------------------------------------------------------------
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
#训练
#迭代次数
max_iters = 600000 # 最大迭代次数
compile = True # 是否使用PyTorch 2.0编译模型以提高速度
# 学习率衰减设置(余弦退火)
learning_rate = 6e-4 # max learning rate
min_lr = 6e-5  # 最小学习率，根据Chinchilla法则约为learning_rate/10
decay_lr = True  # 是否衰减学习率
warmup_iters = 2000  # 预热步数
lr_decay_iters = 600000  # 学习率衰减步数，根据Chinchilla法则应约等于max_iters

# adamW optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

#梯度裁剪
grad_clip = 1.0 # 在训练时将梯度裁剪到这个值，如果设为 0.0 则禁用梯度裁剪

#矩阵乘法精度
matmul_precision='medium' #在'medium','high','highest'中选择

#ddp设置
backend = 'nccl' # 'nccl', 'gloo', etc.
#-----------------------------------------------------------------------------
# 系统
device = 'cuda'  # 设备，如'cpu'、'cuda'、'cuda:0'等，苹果电脑可尝试'mps'
#=============================================================================
#读取配置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(config_path).read())
config = {k: globals()[k] for k in config_keys}
#=============================================================================
#推导必要的参数
out_dir=BASE_DIR.parent /"experiments"/name
iter_num=0 #已经经过的迭代次数

meta_path = data_dir / f"{dataset}"/"meta.json"
with open(meta_path, 'r', encoding='utf-8') as file:
	mata_config = json.load(file)
vocab_size=mata_config['vocab_size']#词典大小
vocab_name=mata_config['vocab_name']#词典名称

torch.set_float32_matmul_precision(matmul_precision)
# 数据类型，若支持bfloat16则使用，否则使用float16，float16会自动使用GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

#ddp
ddp = int(os.environ.get('RANK', -1)) != -1  # 是不是DDP运行？

if ddp:
	init_process_group(backend=backend)  # 初始化进程组
	ddp_rank = int(os.environ['RANK'])  # 全局进程编号（0到world_size-1）
	ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 当前节点内的进程编号（0到节点内GPU数-1）
	ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数（通常=总GPU数）
	device = f'cuda:{ddp_local_rank}'  # 每个进程绑定自己的GPU
	torch.cuda.set_device(device)  # 确保使用指定的GPU
	master_process = ddp_rank == 0  # 主进程（仅1个，负责日志、保存模型等）
	seed_offset = ddp_rank  # 每个进程随机种子不同，避免数据采样重复
	# 调整梯度累积步数（保持总有效batch_size不变）
	assert gradient_accumulation_steps % ddp_world_size == 0
	gradient_accumulation_steps //= ddp_world_size
else:
	master_process = True
	seed_offset = 0
	ddp_world_size = 1
device_type = 'cuda' if 'cuda' in device else 'cpu'
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

if master_process:
	out_dir.mkdir(parents=True, exist_ok=True)  # 创建输出的文件夹


#-----------------------------------------------------------------------------
# #数据读取
if not use_dataloader:
	def get_batch(split):		#获取数据
		if split == 'train':
			data_path = data_dir / f"{dataset}"/"train.npy"
		else:
			data_path = data_dir / f"{dataset}"/"val.npy"
		data = np.memmap(data_path, dtype=np.int32, mode='r')
		ix = torch.randint(len(data) - block_size, (batch_size,))
		x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
		y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
		if device_type == 'cuda':
			# pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
			x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
		else:
			x, y = x.to(device), y.to(device)
		return x, y
else:
	class MyDataset(Dataset):
		def __init__(self, data_path, block_size):
			self.data = np.memmap(data_path, dtype=np.int32, mode='r')  # 内存映射（避免加载全部数据),npy文件
			self.block_size = block_size  # 每个样本的序列长度

		def __len__(self):
			# 样本数量 = 总数据长度 - 序列长度（确保不会越界）
			return len(self.data) - self.block_size

		def __getitem__(self, idx):
			# 根据索引idx获取输入x和目标y（y是x的偏移1位版本）
			x = self.data[idx: idx + self.block_size].astype(np.int64)  # 输入序列
			y = self.data[idx + 1: idx + 1 + self.block_size].astype(np.int64)  # 目标序列（偏移1位）
			return torch.from_numpy(x), torch.from_numpy(y)
	train_data_path = data_dir / f"{dataset}" / "train.npy"
	val_data_path = data_dir / f"{dataset}" / "val.npy"
	train_dataset = MyDataset(train_data_path, block_size=block_size)
	val_dataset = MyDataset(val_data_path, block_size=block_size)
	if ddp:
		# DDP模式：使用分布式采样器（每个进程加载不同数据）
		train_sampler = DistributedSampler(
			train_dataset,
			shuffle=True,  # 训练时打乱数据
			seed=42 + seed_offset  # 每个进程种子不同，避免采样重复
		)
		val_sampler = DistributedSampler(
			val_dataset,
			shuffle=False,  # 验证时不打乱
			seed=42 + seed_offset
		)
	else:
		# 非DDP模式：使用随机采样器
		train_sampler = RandomSampler(train_dataset)
		val_sampler = RandomSampler(val_dataset)
	train_loader=DataLoader(
			train_dataset,
			batch_size=batch_size,  # 微批次大小（与原有batch_size一致）
			sampler=train_sampler,
			num_workers=num_workers,  # 多进程加载（根据CPU核心数调整，建议4-8）
			pin_memory=pin_memory,  # 锁页内存（加速GPU传输）
			drop_last=drop_last,  # 训练时丢弃最后一个不完整批次
			prefetch_factor=prefetch_factor  # 预取2个批次（提前加载数据）
		)
	val_loader=DataLoader(
			val_dataset,
			batch_size=batch_size,  # 微批次大小（与原有batch_size一致）
			sampler=val_sampler,
			num_workers=num_workers,  # 多进程加载（根据CPU核心数调整，建议4-8）
			pin_memory=pin_memory,  # 锁页内存（加速GPU传输）
			drop_last=drop_last,  # 训练时丢弃最后一个不完整批次
			prefetch_factor=prefetch_factor  # 预取2个批次（提前加载数据）
		)
	train_iter = iter(train_loader)
	val_iter = iter(val_loader)


	def get_batch(split):
		global train_iter, val_iter
		if split == 'train':
			try:
				x, y = next(train_iter)  # 从训练迭代器取数据
			except StopIteration:
				# 一个epoch结束后，重新创建迭代器（DDP需要重新设置epoch，确保打乱）
				if ddp:
					train_sampler.set_epoch(iter_num)  # 关键：每个epoch更新种子，确保打乱
				train_iter = iter(train_loader)
				x, y = next(train_iter)
		else:
			try:
				x, y = next(val_iter)
			except StopIteration:
				val_iter = iter(val_loader)
				x, y = next(val_iter)

		# 转移到设备
		if device_type == 'cuda':
			x = x.pin_memory().to(device, non_blocking=True)
			y = y.pin_memory().to(device, non_blocking=True)
		else:
			x = x.to(device)
			y = y.to(device)
		return x, y

#必要的函数
def get_lr(it):		#余弦退火
	if it < warmup_iters:
		return learning_rate * (it + 1) / (warmup_iters + 1)
	if it > lr_decay_iters:
		return min_lr
	decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			with ctx:
				logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out



#-----------------------------------------------------------------------------
# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
				  bias=bias, vocab_size=vocab_size, dropout=dropout,attention_dropout=attention_dropout,RoPE=RoPE)
if args.init_from == "scratch":
	print("从头初始化一个新模型")
	gptconfig=GPTConfig(**model_args)
	model = GPT(gptconfig)
elif args.init_from == 'resume':
	print(f"从{out_dir}重新训练")
	ckpt_path = out_dir/'checkpoint.pt'
	checkpoint = torch.load(ckpt_path, map_location=device)
	checkpoint_model_args = checkpoint['model_args']
	#模型结构与check_point的结构一致
	for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
		model_args[k] = checkpoint_model_args[k]
	gptconf = GPTConfig(**model_args)
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
else:
	pass
model.to(device) #转到GPU

if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])  # 包装模型
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

scaler = torch.amp.GradScaler(device_type,enabled=(dtype == 'float16'))# 混合梯度
if ddp:
	optimizer = model.module.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)# optimizer
else:
	optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# 编译
if compile:
	print("compiling the model... (takes a ~minute)")
	unoptimized_model = model
	model = torch.compile(model) # requires PyTorch 2.0

#wandb log
if wandb_log and master_process and args.init_from == "resume" :
	import wandb
	wandb_project=checkpoint["wandb_project"]
	wandb_run_name=checkpoint["wandb_run_name"]
	wandb_run_id=checkpoint["wandb_run_id"]
	wandb.init(project=wandb_project, name=wandb_run_name, config=config,id=wandb_run_id,resume="must")

if wandb_log and master_process and args.init_from == "scratch" :
	import wandb
	wandb.init(project=wandb_project, name=wandb_run_name, config=config)
#-----------------------------------------------------------------------------
#训练过程
print("Begin training")
start_time=time.time()
X, Y = get_batch('train')  # 第一批数据
t0 = time.time()
raw_model = model.module if ddp else model
local_iter_num = 0  # 当前进程生命周期内的迭代次数
running_mfu = -1.0 #模型浮点运算利用率
best_val_loss=float('inf')
while True:
	if iter_num % eval_interval == 0 and master_process:
		losses = estimate_loss()
		print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
		if wandb_log:
			wandb.log({
				"iter": iter_num,
				"train/loss": losses['train'],
				"val/loss": losses['val'],
				"lr": get_lr(iter_num) if decay_lr else learning_rate,
				"mfu": running_mfu * 100,  # convert to percentage
			})
		if losses['val'] < best_val_loss or always_save_checkpoint:
			best_val_loss = losses['val']
			if iter_num > 0:
				checkpoint = {
					'model': raw_model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'model_args': model_args,
					'iter_num': iter_num,
					'best_val_loss': best_val_loss,
					'config': config,
					"vocab_name": vocab_name,
				}
				if wandb_log and master_process:
					checkpoint['wandb_run_id'] = wandb.run.id  # 唯一标识实验的 ID
					checkpoint['wandb_run_name'] = wandb_run_name  # 实验名称
					checkpoint["wandb_project"]=wandb_project
				print(f"saving checkpoint to {out_dir}")
				torch.save(checkpoint, out_dir/"checkpoint.pt")


	lr = get_lr(iter_num) if decay_lr else learning_rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	for micro_step in range(gradient_accumulation_steps):
		# 前向传播
		if ddp: #在最后一步才通信以同步梯度
			model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
		with ctx:
			logits, loss = model(X, Y)
			loss = loss / gradient_accumulation_steps
		X, Y = get_batch('train')  # 同步数据加载（无异步）,顺序十分重要
		# 反向传播
		scaler.scale(loss).backward()
	if grad_clip > 0:  # grad_clip 是阈值，如 1.0
		scaler.unscale_(optimizer) #梯度裁剪需要特殊处理
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
	scaler.step(optimizer) #更新参数
	scaler.update()
	optimizer.zero_grad(set_to_none=True)

	#记录时间
	t1 = time.time()
	dt = t1 - t0
	t0 = t1
	if iter_num % log_interval == 0:
		# get loss as float. note: this is a CPU-GPU sync point
		elapsed_time = time.time() - start_time  # 已用时间（秒）
		if iter_num > 0:
			avg_time_per_iter = elapsed_time / local_iter_num  # 平均每次迭代耗时（秒）
			remaining_iters = max_iters - iter_num  # 剩余迭代次数
			remaining_seconds = avg_time_per_iter * remaining_iters  # 剩余时间（秒）
			remaining_hours = remaining_seconds / 3600  # 转换为小时
		else:
			remaining_hours = 0.0  # 初始阶段暂不显示
		lossf = loss.item() * gradient_accumulation_steps
		if local_iter_num >= 5:  # 让训练稳定一会儿
			mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
			running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu #EMA
		print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, remaining: {remaining_hours:.2f} hours")
	iter_num += 1
	local_iter_num += 1

	# 结束
	if iter_num > max_iters:
		break
if ddp:
	destroy_process_group()