import argparse
from pathlib import Path
import torch

#=================================================================
name="nanoGPT-2" #要查看的breakpoint对应实验的名称：对应 experiment/ 下的实验数据保存的位置
#=================================================================
#命令行输入处理
parser = argparse.ArgumentParser(description="查看模型")
parser.add_argument("--name", type=str, required=False,default=name,
					help="要查看的模型的名称：对应 experiment/ 下子文件名字")
parser.add_argument("--detail", action="store_true",
					help="是否显示详细信息（如模型结构参数）")
args = parser.parse_args()
#-----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ckpt_path=BASE_DIR.parent /"experiments"/args.name/'checkpoint.pt'
checkpoint = torch.load(ckpt_path)

# 检查文件是否存在
if not ckpt_path.exists():
	print(f"错误：未找到checkpoint文件 {ckpt_path}")
	exit(1)

# 加载checkpoint
try:
	checkpoint = torch.load(ckpt_path, map_location="cpu")  # 加载到CPU，避免GPU内存问题
except Exception as e:
	print(f"加载checkpoint失败：{e}")
	exit(1)

# 打印基本信息
print("=" * 60)
print(f"Checkpoint信息 - 实验名称: {args.name}")
print(f"文件路径: {ckpt_path}")
print("=" * 60)

# 1. 训练进度信息
print("\n[训练进度信息]")
print(f"已完成迭代次数: {checkpoint.get('iter_num', '未知')}")
print(f"最佳验证集损失: {checkpoint.get('best_val_loss', '未知'):.6f}")

# 2. 词汇表信息
print("\n[词汇表信息]")
print(f"使用的词汇表: {checkpoint.get('vocab_name', '未知')}")

# 3. 模型结构参数
print("\n[模型结构参数]")
model_args = checkpoint.get('model_args', {})
if model_args:
	for key, value in model_args.items():
		print(f"  {key}: {value}")
else:
	print("  未找到模型结构参数")

# 4. 训练配置参数
print("\n[训练配置参数]")
config = checkpoint.get('config', {})
if config:
	# 只显示关键配置，避免信息过多
	key_configs = ['n_layer', 'n_head', 'n_embd', 'batch_size',
				   'block_size', 'learning_rate', 'max_iters', 'dropout']
	for key in key_configs:
		if key in config:
			print(f"  {key}: {config[key]}")
	# 如果需要详细信息，通过--detail参数显示
	if args.detail:
		print("\n[详细配置参数]")
		for key, value in config.items():
			print(f"  {key}: {value}")
else:
	print("  未找到训练配置参数")

# 5. 模型参数规模
print("\n[模型参数规模]")
model_state = checkpoint.get('model', {})
if model_state:
	param_count = sum(p.numel() for p in model_state.values())
	print(f"总参数数量: {param_count:,}")
	print(f"参数规模估计: {param_count * 4 / 1024 / 1024:.2f} MB (按float32计算)")

	# 显示部分关键层参数
	if args.detail:
		print("\n[部分参数详情]")
		layer_keys = list(model_state.keys())[:5]  # 只显示前5个参数
		for key in layer_keys:
			param = model_state[key]
			print(f"  {key}: 形状 {param.shape}, 数据类型 {param.dtype}")
		if len(model_state.keys()) > 5:
			print(f"  ... 还有 {len(model_state.keys()) - 5} 个参数未显示")
else:
	print("  未找到模型参数")

# 6. 优化器信息
print("\n[优化器信息]")
optimizer_state = checkpoint.get('optimizer', {})
if optimizer_state:
	print(f"  优化器参数组数: {len(optimizer_state.get('param_groups', []))}")
	if args.detail:
		print(f"  学习率: {optimizer_state['param_groups'][0].get('lr', '未知')}")
else:
	print("  未找到优化器信息")

# 7. wandb实验信息
print("\n[实验跟踪信息]")
if 'wandb_run_id' in checkpoint:
	print(f"  wandb项目: {checkpoint.get('wandb_project', '未知')}")
	print(f"  wandb运行名称: {checkpoint.get('wandb_run_name', '未知')}")
	print(f"  wandb运行ID: {checkpoint['wandb_run_id']}")
else:
	print("  未找到wandb跟踪信息")

print("\n" + "=" * 60)
print("检查完成")