import json
import re
from datetime import datetime

import torch
from torch.nn.utils.rnn import pad_sequence
# scripts/prepare_data.py
import sys
from pathlib import Path
# 将项目根目录添加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import os
import argparse
import pickle
from pathlib import Path
import jieba
from utils.tokenizer_utils import SimpleVocab
import numpy as np
def clean_text(text: str) -> str:
	"""清洗文本，移除冗余符号和空白"""
	# 移除所有换行符、制表符等空白符（保留正常空格）
	text = re.sub(r'[\n\t\r]+', ' ', text)
	# 移除格式符号：{}:;- 等（根据实际情况调整）
	text = re.sub(r'[{}:;\\-]+', '', text)
	# 合并连续空格为单个空格
	text = re.sub(r' +', ' ', text).strip()
	return text

def prepare_dataset(raw_data_dir: Path,vocab_dir, output_dir: Path,block_size:int,val_ratio:float=0.1,vocab_name: str="sci-fi"):
	"""
	预处理单个数据集：
	1. 遍历 raw_data_dir 下的所有文本文件
	2. 用 tokenizer_type 分词（可切换：GPT2Tokenizer、jieba等）
	3. 保存为统一格式（如：token_ids.npy + meta.json）
	"""
	with open(vocab_dir, 'rb') as f:
		vocab = pickle.load(f)
	all_idx = []
	for text_file in Path(raw_data_dir).rglob("*"):
		if text_file.is_file() and text_file.name.endswith(".json"):
			with open(text_file, 'r', encoding='utf-8') as f:
				for line in f:
					obj = json.loads(line)
					text=clean_text(obj['text'])
					idx = vocab.encode(jieba.cut(text))
					all_idx.append(idx)
					print(idx)
					print(text)
					print(list(jieba.cut(text)))
		elif text_file.is_file() and text_file.name.endswith(".txt"):
			try:
				with open(text_file, 'r', encoding='utf-8') as f:
					for line in f:
						text = line.strip()
						if not text:
							continue
						# 清洗+分词
						#idx = vocab.encode(jieba.cut(text))
						idx = vocab.encode(list(text))
						all_idx.append(idx)
			except Exception as e:
				print(f"跳过{text_file}:{e}")
	all_tokens=[]
	for idx in all_idx:
		all_tokens.extend(idx)
	# 3. 保存预处理后的数据
	all_tokens=torch.tensor(all_tokens, dtype=torch.int32)
	print(f"共有{len(all_tokens)}个tokens")
	num_blocks = len(all_tokens) // block_size
	blocks = [all_tokens[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]

	# 确定验证块的数量
	num_val_blocks = int(num_blocks * val_ratio)

	# 间隔选择验证块（例如每5块选1块）
	val_indices = np.linspace(0, num_blocks - 1, num_val_blocks, dtype=int)

	# 划分训练集和验证集
	train_blocks = [block for i, block in enumerate(blocks) if i not in val_indices]
	val_blocks = [block for i, block in enumerate(blocks) if i in val_indices]

	# 拼接回完整序列
	train_tokens = np.concatenate(train_blocks)
	val_tokens = np.concatenate(val_blocks)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	np.save(output_dir / "train.npy", np.array(train_tokens, dtype=np.int32))
	np.save(output_dir / "val.npy", np.array(val_tokens, dtype=np.int32))

	with open(output_dir / "meta.json", "w") as f:
		json.dump({
			"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			"total_tokens": len(all_tokens),
			"train_tokens": len(train_tokens),
			"val_tokens": len(val_tokens),
			"val_ratio": val_ratio,
			"vocab_size": len(vocab),
			"vocab_name": vocab_name
		}, f)

#python scripts/prepare_data.py --dataset wiki_zh --vocab wiki_zh.pkl --output wiki
#python scripts/prepare_data.py --dataset sci-fi  --vocab sci-fi.pkl --output sci-fi
if __name__ == "__main__":
	BASE_DIR = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, required=True,
	                    help="数据集名称：对应 raw/ 下的子目录（如 wiki_zh.plt、shakespeare）")
	parser.add_argument("--vocab", type=str, required=True,
	                    help="词典名称：对应 vocab/ 下的pkl文件")
	parser.add_argument("--output", type=str, required=True,
	                    help="处理后的数据储存的位置")
	parser.add_argument("--block_size", type=int, required=False,default=8192,
	                    help="交叉划分时块的大小")
	parser.add_argument("--val_ratio", type=float, required=False,default=0.1,
	                    help="验证集的比例")
	args = parser.parse_args()

	raw_data_dir =BASE_DIR.parent/"data"/"raw"/f"{args.dataset}"
	vocab_dir = BASE_DIR.parent / "data" / "vocab" / f"{args.vocab}"
	output_dir = BASE_DIR.parent/"data"/"processed"/f"{args.output}"

	prepare_dataset(raw_data_dir, vocab_dir,output_dir, args.block_size, args.val_ratio,args.vocab)