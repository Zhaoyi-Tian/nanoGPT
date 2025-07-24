
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + 1e-6)
        if self.bias is not None:
            return x*self.weight + self.bias
        else:
            return x*self.weight


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn=nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout1 = nn.Dropout(config.attention_dropout)
        self.dropout2 = nn.Dropout(config.attention_dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout= config.attention_dropout
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.hs = config.n_embd // config.n_head  # 每个注意力头的维度
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(self.hs, config.block_size))
        self.config = config


    def _precompute_freqs_cis(self, dim, end, theta=10000.0):
        """预计算复数域的旋转频率（RoPE核心）"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
        t = torch.arange(end, device=freqs.device)  # 位置索引
        freqs = torch.outer(t, freqs).float()  # (end, dim//2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数表示 (end, dim//2)
        return freqs_cis

    def _apply_rotary_emb(self, x, freqs_cis):
        """对输入向量x（Q或K）应用旋转编码"""
        # x形状：(B, n_head, T, hs)，hs为每个头的维度
        B, n_head, T, hs = x.shape
        # 将x拆分为实部和虚部（假设hs为偶数）
        x = x.view(B, n_head, T, hs // 2, 2)  # (B, n_head, T, hs//2, 2)
        x = x.to(torch.float32)
        x = torch.view_as_complex(x)  # 转换为复数 (B, n_head, T, hs//2)
        # 应用旋转：x * freqs_cis（广播机制）
        freqs_cis = freqs_cis[:T].view(1, 1, T, hs // 2)  # 适配维度
        x_rot = x * freqs_cis
        x_out = torch.view_as_real(x_rot).flatten(3)  # 转回实数并合并维度
        x_out=x_out.to(torch.bfloat16)
        return x_out


    def forward(self, x):
        B, T, C = x.size()
        x=self.c_attn(x)
        Q,K,V=x.split(self.n_embd, -1)
        Q=Q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)#(B, n_head, T, hs)
        K=K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, hs)
        V=V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, hs)
        if self.config.RoPE:  # 需要在config中传入RoPE参数
            Q = self._apply_rotary_emb(Q, self.freqs_cis)
            K = self._apply_rotary_emb(K, self.freqs_cis)
        y=torch.nn.functional.scaled_dot_product_attention(Q,K,V,attn_mask=None, dropout_p=self.dropout,is_causal=True)
        # attention=Q@K.transpose(2, 3)* (1.0 / math.sqrt(K.size(-1))) #(B,n_head,T,T)
        # attention = attention.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        # attention = F.softmax(attention, dim=-1) #(B,n_head,T,T)
        # attention = self.dropout1(attention)
        # y = attention @ V  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y=self.c_proj(y)
        y=self.dropout2(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 4* config.n_embd, bias=config.bias)
        self.dropout1 = nn.Dropout(config.attention_dropout)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout2 = nn.Dropout(config.attention_dropout)
        self.glue=nn.GELU()
    def forward(self, x):
        y=self.c_attn(x)
        y=self.glue(y)
        y=self.dropout1(y)
        y=self.c_proj(y)
        y=self.dropout2(y)
        return y


class block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x=self.attn(self.ln_1(x))+x
        x=self.mlp(self.ln_2(x))+x
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #序列最大长度
    vocab_size: int = 50304 # 词典大小
    n_layer: int = 12 #Transformer 层数
    n_head: int = 12 #注意力头数
    n_embd: int = 768 #嵌入维度
    dropout: float = 0.0 #dropout
    attention_dropout: float = 0.0 #attention_dropout
    bias: bool = True #是否采用偏置
    RoPE: bool = True #是否采用旋转编码

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not self.config.RoPE:
            self.transformer =nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置编码
            drop = nn.Dropout(config.dropout),  # Dropout层
            h = nn.ModuleList([block(config) for _ in range(config.n_layer)]),  # Transformer块列表
            ln_f = LayerNorm(config.n_embd, bias=config.bias),  # 最终层归一化
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),  # Dropout层
                h=nn.ModuleList([block(config) for _ in range(config.n_layer)]),  # Transformer块列表
                ln_f=LayerNorm(config.n_embd, bias=config.bias),  # 最终层归一化
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.weight=self.lm_head.weight #参数共享
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)) #TODO:关键优化
        print(f"参数量为{self.get_num_params()/1e6}M")



    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # (t)

        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        if not self.config.RoPE:
            pos_emb = self.transformer.wpe(pos) # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x= self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 训练阶段
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段
            logits = self.lm_head(x[:, [-1], :]) #用最后一个
            loss = None

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        #生成文字
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"采用L2正则化的参数数量:{num_decay_params}")
        print(f"未采用L2正则化的参数数量:{num_nodecay_params}")
        extra_args={}
        if device_type=="cuda":
            extra_args={"fused": True}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {device_type=='cuda'}")
        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        #以A100芯片为例计算浮点运算利用率
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


