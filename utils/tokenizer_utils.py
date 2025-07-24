import json
import pickle
from pathlib import Path

import jieba
from collections import Counter
from typing import List, Dict, Set, Optional
import re


def clean_text(text: str) -> str:
    """清洗文本，保留中文、英文、数字和常用标点，移除其他 Unicode 字符"""
    # 移除版权声明块（跨多行）
    text = re.sub(r'={5,}.*?={5,}', '', text, flags=re.DOTALL)

    # 将全角空格(\u3000)替换为普通空格
    text = text.replace('\u3000', ' ')

    # 保留中文、英文、数字和常用标点，移除其他字符
    # 常用标点包括：，。！？；：“”‘’（）【】
    text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z\s，。！？；：“”‘’（）【】]', '', text)

    # 移除所有换行符、制表符等空白符（保留正常空格）
    text = re.sub(r'[\n\t\r]+', ' ', text)
    # 合并连续空格为单个空格
    text = re.sub(r' +', ' ', text).strip()
    return text

class SimpleVocab:
    def __init__(self,
                 special_tokens: List[str] = None,
                 min_freq: int = 1,
                 max_size: Optional[int] = None):
        """
        初始化简单词表
        参数:
            special_tokens: 特殊token列表，如["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            min_freq: 最小词频，低于此频率的词将被忽略
            max_size: 词表最大大小，None表示不限制
        """
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>"]
        self.min_freq = min_freq
        self.max_size = max_size
        # 词到ID的映射
        self.word2id = {}
        # ID到词的映射
        self.id2word = {}
        # 词频统计
        self.word_freq = Counter()
        # 初始化特殊token
        self._init_special_tokens()

    def _init_special_tokens(self):
        """初始化特殊token"""
        for token in self.special_tokens:
            self.word2id[token] = len(self.word2id)
            self.id2word[len(self.id2word)] = token

    def read_text(self, texts: List[List[str]]):
        """
        从文本构建词表

        参数:
            texts: 文本列表，每个文本是token列表
        """
        # 统计词频
        for tokens in texts:
            self.word_freq.update(tokens)

    # 按词频排序
    def build_vocab(self,path):
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: (-x[1], x[0])  # 先按词频降序，再按字典序升序
        )
        # while True:
        #     self.min_freq=int(input())
        #     if self.min_freq == -1:
        #         break
        #     # 构建词表
        #     for word, freq in sorted_words:
        #         # 跳过特殊token
        #         if word in self.special_tokens:
        #             continue
        #         if word[0]=="/":
        #             continue
        #         if word[0]=="\\":
        #             continue
        #         # 检查最小词频
        #         if freq < self.min_freq:
        #             break
        #         # 检查最大词表大小
        #         if self.max_size is not None and len(self.word2id) >= self.max_size:
        #             break
        #
        #         # 添加到词表
        #         self.word2id[word] = len(self.word2id)
        #         self.id2word[len(self.id2word)] = word
        #     with open(path, 'wb') as f:
        #         pickle.dump(self, f)
        #     print(len(self.word2id))
        #     print(self.word2id)
        #     self.word2id.clear()
        #     self.id2word.clear()
        #     self._init_special_tokens()
        for word, freq in sorted_words:
            # 跳过特殊token
            if word in self.special_tokens:
                continue
            if word[0]=="/":
                continue
            if word[0]=="\\":
                continue
            # 检查最小词频
            if freq < self.min_freq:
                break
            # 检查最大词表大小
            if self.max_size is not None and len(self.word2id) >= self.max_size:
                break
            # 添加到词表
            self.word2id[word] = len(self.word2id)
            self.id2word[len(self.id2word)] = word
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(self.word2id)


    def encode(self, tokens: List[str]) -> List[int]:
        """将token列表转换为ID列表"""
        return [self.word2id.get(token, self.word2id["<UNK>"]) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """将ID列表转换为token列表"""
        return [self.id2word.get(idx, "<UNK>") for idx in ids]

    def __len__(self):
        """返回词表大小"""
        return len(self.word2id)

    def __contains__(self, word: str):
        """检查词是否在词表中"""
        return word in self.word2id


# 示例用法
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent
    # 构建wiki的词表
    # vocab = SimpleVocab(
    #     special_tokens=["<PAD>", "<UNK>"],
    #     min_freq=20,
    #     max_size=None
    # )
    # count=0
    # for text_file in Path(BASE_DIR.parent/"data"/"raw"/"wiki_zh").rglob("*"):
    #     if text_file.is_file():
    #         with open(text_file, 'r', encoding='utf-8') as f:
    #             for line in f:
    #                 obj=json.loads(line)
    #                 text=clean_text(obj['text'])
    #                 vocab.read_text([list(jieba.cut(text))])
    #         count+=1
    #         print(f"wiki: {count}")
    # vocab.build_vocab(BASE_DIR.parent/"data"/"vocab"/"wiki_zh.pkl")

    # 构建sci-fi的词表
    vocab = SimpleVocab(
        special_tokens=["<PAD>", "<UNK>"],
        min_freq=35,
        max_size=None
    )
    batch_size = 100
    count = 0  # 用于统计处理的文件数

    for text_file in Path(BASE_DIR.parent / "data" / "raw" / "sci-fi").rglob("*"):
        if text_file.is_file():
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    # 按块读取整个文件内容
                    content = f.read()
                    # 移除版权声明块（跨多行）
                    cleaned_content = clean_text(content)
                    # 按段落分割（假设段落间用空行分隔）
                    paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if p.strip()]

                    batch_paras = []  # 临时缓存批次段落

                    for para in paragraphs:
                        if not para:
                            continue
                        # 分词处理,jieba
                        # tokens = list(jieba.cut(para, cut_all=False, HMM=True))
                        # batch_paras.append(tokens)
                        #因算力限制，故采用字符级分词，更快收敛
                        tokens = list(para)
                        batch_paras.append(tokens)

                        # 达到批次大小，更新词频并清空缓存
                        if len(batch_paras) >= batch_size:
                            vocab.read_text(batch_paras)
                            batch_paras = []  # 清空缓存，释放内存

                    # 处理剩余不足一个批次的段落
                    if batch_paras:
                        vocab.read_text(batch_paras)

                count += 1
                print(f"成功处理文件: {text_file}, 文件计数: {count}")

            except UnicodeDecodeError:
                print(f"跳过文件: {text_file} - 无法使用utf-8编码读取")
            except Exception as e:
                print(f"跳过文件: {text_file} - 发生未知错误: {e}")

    vocab.build_vocab(BASE_DIR.parent / "data" / "vocab" / "sci-fi.pkl")
    print(f"词表大小: {len(vocab)}")
