#-----------------------------------------------------------------------------
#实验基本情况
name="nano-GPT" #实验的名称：对应 experiment/ 下的实验数据保存的位置,如果续训的话要保证与原来的实验名称一致
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