import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width=512,
        cond_height=512,
        number=0,
        n_loras=1
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        #### img condition
        batch_size = hidden_states.shape[0]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size =  hidden_states.shape[1] - cond_size * self.n_loras
        shape = (batch_size, hidden_states.shape[1], 3072)
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype) 
        mask[:, :block_size+self.number*cond_size, :] = 0
        mask[:, block_size+(self.number+1)*cond_size:, :] = 0
        hidden_states = mask * hidden_states
        ####
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

# 新定义的类，基底是MultiDoubleStreamBlockLoraProcessor
# 
# 

class CombinedAttnProcessor_double(nn.Module):
    """
    结合了IPAFluxAttnProcessor2_0与MultiDoubleStreamBlockLoraProcessor两个类功能的处理器。
    
    主要特点：
      - 采用第二个类的整体框架（如对 encoder_hidden_states 的 context 投影、LoRA 层附加、masked attention 计算等）
      - 同时支持 image_emb 分支的处理（类似 IPAFluxAttnProcessor2_0），通过额外投影获得 image 的 key/value 并计算额外的 attention，
        最后将该结果按 scale 加到主分支上。
    
    参数:
      dim: 输入的维度，一般与 hidden_size 相同
      ranks, lora_weights, network_alphas, device, dtype, cond_width, cond_height, n_loras:
             同 MultiDoubleStreamBlockLoraProcessor 定义，控制 LoRA 部分
      hidden_size: 用于 image_emb 分支的尺寸（默认取 dim）
      cross_attention_dim: image_emb 分支中输入到投影层的维度（默认与 hidden_size 相同）
      scale: image_emb 分支输出添加到主 attention 输出前的缩放因子
      num_tokens: （可选）对应 IPAFluxAttnProcessor2_0 中的参数，目前未在其他部分使用
    """
    def __init__(
        self, 
        dim: int, 
        ranks=[], 
        lora_weights=[], 
        network_alphas=[], 
        device=None, 
        dtype=None, 
        cond_width: int = 512, 
        cond_height: int = 512, 
        n_loras: int = 1,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 4
    ):
        super().__init__()
        # --- LoRA 部分，与 MultiDoubleStreamBlockLoraProcessor 类似 ---
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, 
                              cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, 
                              cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, 
                              cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, 
                              cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        
        # --- image_emb 分支部分，参考 IPAFluxAttnProcessor2_0 ---
        # 如果未给出 hidden_size 则取 dim
        if hidden_size is None:
            hidden_size = dim
        self.hidden_size = hidden_size
        # 如果 cross_attention_dim 未指定，则与 hidden_size 相同
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else hidden_size
        self.scale = scale
        self.num_tokens = num_tokens
        
        # 投影 image_emb 到 key 和 value 的空间
        self.to_k_ip = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)
        self.to_v_ip = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False)
        # 采用 RMSNorm 对投影后的 key 进行归一化
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        
    def __call__(
        self,
        attn:Attention,  # Attention 模块，应包含如 to_q, to_k, to_v, add_q_proj, add_k_proj, add_v_proj, norm_q, norm_k, norm_added_q, norm_added_k, to_out, to_add_out 等成员
        hidden_states: torch.FloatTensor,
        image_emb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
        causal_attn: bool = True
    ) -> torch.FloatTensor:
        """
        执行融合后的 attention 计算。
        
        如果 encoder_hidden_states 不为 None，则进行 context 投影后与当前 hidden_states 拼接；
        如果 image_emb 不为 None，则执行额外的 image 分支 attention，并在最后将其结果按 scale 相加。
        
        返回值根据 use_cond 参数返回不同格式：
          - use_cond=True 时返回 (主流 hidden_states, encoder_hidden_states, cond_hidden_states)
          - 否则返回 (encoder_hidden_states, 主流 hidden_states)
          - 若 encoder_hidden_states 为空，则返回最终的 hidden_states
        """
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        
        # --- 1. Context（encoder_hidden_states）投影（如果提供） ---
        if encoder_hidden_states is not None:
            # 采用 attn 中的 context 投影层
            enc_q_proj = attn.add_q_proj(encoder_hidden_states)
            enc_k_proj = attn.add_k_proj(encoder_hidden_states)
            enc_v_proj = attn.add_v_proj(encoder_hidden_states)
            # 假设 attn.heads 已存在，同时 head_dim 可从投影后的 tensor 确定
            # 此处与第二个类中写法保持一致（第二类中有固定 inner_dim = 3072 的写法，此处以 tensor shape 为准）
            inner_dim = enc_k_proj.shape[-1]
            head_dim = inner_dim // attn.heads
            enc_q_proj = enc_q_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k_proj = enc_k_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v_proj = enc_v_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                enc_q_proj = attn.norm_added_q(enc_q_proj)
            if attn.norm_added_k is not None:
                enc_k_proj = attn.norm_added_k(enc_k_proj)
        # ---------------------------
        
        # --- 2. 主分支 hidden_states 投影并加上 LoRA 修正 ---
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key   = key   + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)
        
        # 确定 head_dim（假设 attn.heads 已定义）
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # --- 3. image_emb 分支的处理（如果提供）---
        if image_emb is not None:
            ip_hidden_states = image_emb
            ip_key_proj = self.to_k_ip(ip_hidden_states)
            ip_val_proj = self.to_v_ip(ip_hidden_states)
            ip_key_proj = ip_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_val_proj = ip_val_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_key_proj = self.norm_added_k(ip_key_proj)
            # 采用 scaled dot product attention 计算 image 分支输出
            ip_out = F.scaled_dot_product_attention(
                query, ip_key_proj, ip_val_proj, dropout_p=0.0, is_causal=False
            )
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_out = ip_out.to(query.dtype)
        else:
            ip_out = None
        
        # --- 4. 拼接 context 投影 ---
        if encoder_hidden_states is not None:
            # 将 context 投影与主分支拼接
            query = torch.cat([enc_q_proj, query], dim=2)
            key = torch.cat([enc_k_proj, key], dim=2)
            value = torch.cat([enc_v_proj, value], dim=2)
        
        # --- 5. 可选：应用 rotary embedding（如果提供） ---
        # if image_rotary_emb is not None:
        #     from diffusers.models.embeddings import apply_rotary_emb
        #     query = apply_rotary_emb(query, image_rotary_emb)
        #     key = apply_rotary_emb(key, image_rotary_emb)
        
        # --- 6. 生成 causal mask（参考第二个类） ---
        cond_size = (self.cond_width // 8) * (self.cond_height // 8) * 16 // 64
        # 这里 block_size 基于原始 hidden_states 的序列长度
        block_size = hidden_states.shape[1] - cond_size * self.n_loras
        scaled_seq_len = query.shape[2]
        scaled_block_size = scaled_seq_len - cond_size * self.n_loras
        
        if causal_attn:
            num_cond_blocks = self.n_loras
            mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
            mask[:scaled_block_size, :] = 0  # 前 block_size 行置 0
            for i in range(num_cond_blocks):
                start = i * cond_size + scaled_block_size
                end = (i + 1) * cond_size + scaled_block_size
                mask[start:end, start:end] = 0  # 对角线区域置 0
            mask = mask * -1e20
            mask = mask.to(query.dtype)
        else:
            mask = None
        
        # --- 7. 主 attention 计算 ---
        attn_out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        attn_out = attn_out.to(query.dtype)
        
        # --- 8. 根据是否有 encoder_hidden_states 分离输出 ---
        if encoder_hidden_states is not None:
            enc_seq_len = encoder_hidden_states.shape[1]
            encoder_out = attn_out[:, :enc_seq_len, :]
            main_out = attn_out[:, enc_seq_len:, :]
        else:
            main_out = attn_out
            encoder_out = None
        
        # --- 9. 将 image_emb 分支的输出添加到主分支 ---
        if ip_out is not None:
            main_out = main_out + self.scale * ip_out
        
        # --- 10. 输出投影及 LoRA 后处理 ---
        main_out = attn.to_out[0](main_out)
        for i in range(self.n_loras):
            main_out = main_out + self.lora_weights[i] * self.proj_loras[i](main_out)
        main_out = attn.to_out[1](main_out)
        
        if encoder_out is not None:
            encoder_out = attn.to_add_out(encoder_out)
            # 根据第二个类，将 main_out 分为 block 输出和 cond 输出
            cond_hidden_states = main_out[:, block_size:, :]
            main_out = main_out[:, :block_size, :]
            return (main_out, encoder_out, cond_hidden_states) if use_cond else (encoder_out, main_out)
        else:
            return main_out



# 假定 RMSNorm、LoRALinearLayer 已经在其他地方定义并导入
# 例如：
# from some_module import RMSNorm, LoRALinearLayer

class CombinedAttnProcessor_single(nn.Module):
    """
    合并了 IPAFluxAttnProcessor2_0 与 MultiSingleStreamBlockLoraProcessor 功能的注意力处理器，
    支持：
      - 对 hidden_states 计算 query, key, value（包含 LoRA 调整）
      - 如提供 image_emb，则通过 ip-adapter 分支进行附加投影并对注意力输出做加权补充
      - 如提供 encoder_hidden_states，则利用 context 投影拼接 encoder 信息，再在注意力后分离 encoder 部分和主分支，
        并对主分支进行线性与 dropout 投影（调用 attn.to_out 与 attn.to_add_out）
      - 如未提供 encoder_hidden_states 但 use_cond=True，则按 cond_width/cond_height 和 n_loras 计算条件 token 数量，
        构造遮罩（可选 causal_attn）并在注意力后将输出分为主输出和条件输出
    """
    def __init__(
        self,
        dim: int,  # 使用 dim 参数
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 4,
        ranks: list = [],
        lora_weights: list = [],
        network_alphas: list = [],
        device=None,
        dtype=None,
        cond_width: int = 512,
        cond_height: int = 512,
        n_loras: int = 1,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim  # 将 dim 参数作为类的一个属性
        self.hidden_size = dim  # 使用 dim 作为 hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        
        # LoRA 层：分别用于 Q, K, V 的微调（参数 lora_weights 用于缩放，n_loras 是 LoRA 层数）
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.lora_weights = lora_weights
        
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        
        # ip-adapter 层（来自 IPAFluxAttnProcessor2_0）
        self.to_k_ip = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        # 此处 128 通常与 head_dim 对应，可根据实际情况调整
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn:Attention,
        hidden_states: torch.FloatTensor,
        image_emb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
        causal_attn: bool = True
    ) -> torch.FloatTensor:
        """
        参数说明：
          - attn：包含必要投影层（to_q, to_k, to_v）、归一化层（norm_q, norm_k）以及后处理层（to_out, to_add_out）
          - hidden_states：主序列输入
          - image_emb：当不为 None 时，利用 ip-adapter 分支进行额外的注意力计算
          - encoder_hidden_states：额外上下文信息，用于 context attention 分支（如果不为 None，则优先使用此分支）
          - attention_mask：当前未使用，可扩展传入
          - image_rotary_emb：旋转位置编码，如不为 None，则用于 Q 与 K 的旋转
          - use_cond：如果无 encoder_hidden_states 但需要条件输出，则设置为 True，输出将分为主输出与条件输出
          - causal_attn：在 use_cond 分支中是否构造因果遮罩
        """
        # 若 encoder_hidden_states 存在，则按其批次大小获取 batch_size，否则从 hidden_states 获取
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        
        # 1. 基本的 Q, K, V 投影
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        # 2. 添加 LoRA 调整（对 hidden_states 进行微调后加到 Q, K, V 上）
        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # 3. 重塑 Q, K, V 到 (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # 4. 如果提供 image_rotary_emb，则对 Q 和 K 应用旋转位置编码
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # 5. ip-adapter 分支：如果提供 image_emb，则计算额外的注意力输出
        if image_emb is not None:
            ip_hidden = image_emb
            ip_hidden_key = self.to_k_ip(ip_hidden)
            ip_hidden_value = self.to_v_ip(ip_hidden)
            ip_hidden_key = ip_hidden_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_hidden_value = ip_hidden_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_hidden_key = self.norm_added_k(ip_hidden_key)
            ip_attn = F.scaled_dot_product_attention(query, ip_hidden_key, ip_hidden_value,
                                                     dropout_p=0.0, is_causal=False)
            ip_attn = ip_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_attn = ip_attn.to(query.dtype)
        # 6. 根据情况选择分支：
        # （1）如果 encoder_hidden_states 存在，使用 context 投影
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)
            
            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            
            # 拼接 context 投影与主投影：按 token 维度拼接
            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)
            
            mask = None  # 此分支下不使用额外遮罩
        
        if causal_attn:
            # 根据 cond_width 和 cond_height 计算条件 token 数量（与 second 类保持一致）
            cond_size = (self.cond_width // 8) * (self.cond_height // 8) * 16 // 64
            # 假设输入 hidden_states 的 token 数量中后面部分为条件部分
            block_size = hidden_states.shape[1] - cond_size * self.n_loras
            scaled_seq_len = query.shape[2]
            scaled_block_size = block_size
            scaled_cond_size = cond_size
            mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
            mask[:scaled_block_size, :] = 0
            for i in range(self.n_loras):
                start = i * scaled_cond_size + scaled_block_size
                end = (i + 1) * scaled_cond_size + scaled_block_size
                mask[start:end, start:end] = 0
            mask = mask * -1e20
            mask = mask.to(query.dtype)
        else:
            mask = None
        
        # 7. 进行注意力计算
        attn_out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        
        # 8. 后处理：不同分支下对注意力输出进行不同处理
        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            # 前 enc_len 个 token 为 context 输出，剩下为主输出
            enc_out = attn_out[:, :enc_len, :]
            main_out = attn_out[:, enc_len:, :]
            if image_emb is not None:
                main_out = main_out + self.scale * ip_attn
            # 利用 attn.to_out[0] 与 attn.to_out[1] 进行主输出的线性与 dropout 投影
            main_out = attn.to_out[0](main_out)
            main_out = attn.to_out[1](main_out)
            # 对 context 输出调用 attn.to_add_out
            enc_out = attn.to_add_out(enc_out)
            return main_out, enc_out
        else:
            if image_emb is not None:
                attn_out = attn_out + self.scale * ip_attn
            return attn_out