
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 50304 
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Implement the CausalSelfAttention class
        # Attributes that could possibly be used: config.n_embd, config.n_head, config.dropout, config.bias
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1) 共享线性映射得到 Q,K,V （带偏置）
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)

        # 7) 输出投影
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)

        # dropout
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # 4) 预先构造最多 block_size × block_size 的下三角 Mask
        # shape = (1, 1, L, L) 方便直接 broadcast 到 (B, h, L, L)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0)
        # register_buffer 使其跟随模型移动到 GPU/CPU，但不参与梯度
        self.register_buffer("causal_mask", mask)
        

    def forward(self, x):
        # shape of x: B, L, C
        # shape of output: B, L, C
        # TODO: Implement the CausalSelfAttention class
        B, L, C = x.shape  # B=batch, L=seq_len, C=embedding_dim

        # 1) 得到 Q,K,V  —— 形状 (B, L, 3C) → 分割成三个 (B, L, C)
        qkv = self.qkv_proj(x)  # (B, L, 3C)
        q, k, v = qkv.chunk(3, dim = -1)  # 各 (B, L, C)

        # 2) reshape 成多头 (B, h, L, d)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, L, d)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, L, d)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, L, d)

        # 3) 缩放点积注意力: A = Q K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, L, L)

        # 4) 因果 Mask：在对角线以上的位置填充 -inf
        causal_mask = self.causal_mask[:, :, :L, :L]  # 与当前序列长度对齐
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))

        # 5) Softmax
        attn = F.softmax(attn, dim = -1)  # (B, h, L, L)
        attn = self.attn_drop(attn)

        # 6) 计算上下文表示 Z = A V
        z = attn @ v  # (B, h, L, d)

        # 7) 合并头并输出 Projection
        z = z.transpose(1, 2).contiguous().view(B, L, C)  # (B, L, C)
        y = self.out_proj(z)
        y = self.resid_drop(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
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
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: Tensor of shape (B, T)
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering (int)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_top_p(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """
        Generate text using top-k and/or top-p (nucleus) sampling.

        Args:
            idx: Tensor of shape (B, T)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (int)
            top_p: top-p (nucleus) sampling (float, in [0, 1])
        """
        # TODO: Implement text generation with top-p (nucleus) sampling.
        # top_p: top-p (nucleus) sampling (float, in [0, 1])
        for _ in range(max_new_tokens):
            # 如果序列太长则截断
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 获取模型预测
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # 计算概率
            probs = F.softmax(logits, dim = -1)

            # top-p 采样
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, dim = -1, descending = True)
                cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

                # 创建mask: 移除累积概率超过p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 确保至少保留一个token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # 将不需要的token概率设为0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim = 1, index = sorted_indices, src = sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)

                # 重新归一化概率
                if torch.any(probs > 0):
                    probs = probs / probs.sum(dim = -1, keepdim = True)
                else:
                    # 如果所有token都被移除了，回退到top-1
                    probs = torch.zeros_like(probs)
                    probs[..., torch.argmax(sorted_probs)] = 1.0

            # 从处理后的分布中采样
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx