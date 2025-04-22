from diffusers.models.attention_processor import FluxAttnProcessor2_0
from safetensors import safe_open
import re
import torch
from src.IPA_canny_layer import CombinedAttnProcessor_double,CombinedAttnProcessor_single

device = "cuda"

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            print(f"Loaded tensor {key} with shape {tensors[key].shape}")
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    return checkpoint

def update_model_with_lora(checkpoint, lora_weights, transformer, cond_size):
        number = len(lora_weights)
        ranks = [get_lora_rank(checkpoint) for _ in range(number)]
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                            lora_state_dicts[key] = value
                
                print("setting LoRA Processor for", name)
                # lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                #     dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number
                # )
                lora_attn_procs[name] = CombinedAttnProcessor_double(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number ,hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=128)##todo：加上num_tokens=128
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.down.weight', None)
                    lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.up.weight', None)
                    lora_attn_procs[name].to(device)
                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[key] = value
                
                print("setting LoRA Processor for", name)
                # lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                #     dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number
                # )
                lora_attn_procs[name] = CombinedAttnProcessor_single(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number,hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=128)

                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].to(device)
            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()

        transformer.set_attn_processor(lora_attn_procs)
        

def update_model_with_multi_lora(checkpoints, lora_weights, transformer, cond_size,ip_adapter_path):
        image_proj_model = MLPProjModel(
            cross_attention_dim=4096, # 4096
            id_embeddings_dim=1152, 
            num_tokens=128,
        ).to(device=device, dtype=torch.bfloat16)


        ck_number = len(checkpoints)
        cond_lora_number = [len(ls) for ls in lora_weights]
        cond_number = sum(cond_lora_number)
        ranks = [get_lora_rank(checkpoint) for checkpoint in checkpoints]
        multi_lora_weight = []
        for ls in lora_weights:
            for n in ls:
                multi_lora_weight.append(n)
        
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                lora_state_dicts = [{} for _ in range(ck_number)]
                for idx, checkpoint in enumerate(checkpoints):
                    for key, value in checkpoint.items():
                        # Match based on the layer index in the key (assuming the key contains layer index)
                        if re.search(r'\.(\d+)\.', key):
                            checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                            if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                                lora_state_dicts[idx][key] = value
                
                print("setting LoRA Processor for", name)
                # lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                #     dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=multi_lora_weight, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number
                # )
                lora_attn_procs[name] = CombinedAttnProcessor_double(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number ,hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=128)
                # Load the weights from the checkpoint dictionary into the corresponding layers
                num = 0
                for idx in range(ck_number):
                    for n in range(cond_lora_number[idx]):
                        lora_attn_procs[name].q_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.down.weight', None)
                        lora_attn_procs[name].q_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.up.weight', None)
                        lora_attn_procs[name].k_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.down.weight', None)
                        lora_attn_procs[name].k_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.up.weight', None)
                        lora_attn_procs[name].v_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.down.weight', None)
                        lora_attn_procs[name].v_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.up.weight', None)
                        lora_attn_procs[name].proj_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.proj_loras.{n}.down.weight', None)
                        lora_attn_procs[name].proj_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.proj_loras.{n}.up.weight', None)
                        lora_attn_procs[name].to(device)
                        num += 1
                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = [{} for _ in range(ck_number)]
                for idx, checkpoint in enumerate(checkpoints):
                    for key, value in checkpoint.items():
                        # Match based on the layer index in the key (assuming the key contains layer index)
                        if re.search(r'\.(\d+)\.', key):
                            checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                            if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                                lora_state_dicts[idx][key] = value
                
                print("setting LoRA Processor for", name)
                # lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                #     dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=multi_lora_weight, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number
                # )
                lora_attn_procs[name] = CombinedAttnProcessor_single(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number,hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim,
                num_tokens=128)
                # Load the weights from the checkpoint dictionary into the corresponding layers
                num = 0
                for idx in range(ck_number):
                    for n in range(cond_lora_number[idx]):
                        lora_attn_procs[name].q_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.down.weight', None)
                        lora_attn_procs[name].q_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.q_loras.{n}.up.weight', None)
                        lora_attn_procs[name].k_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.down.weight', None)
                        lora_attn_procs[name].k_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.k_loras.{n}.up.weight', None)
                        lora_attn_procs[name].v_loras[num].down.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.down.weight', None)
                        lora_attn_procs[name].v_loras[num].up.weight.data = lora_state_dicts[idx].get(f'{name}.v_loras.{n}.up.weight', None)
                        lora_attn_procs[name].to(device)
                        num += 1

            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()

        transformer.set_attn_processor(lora_attn_procs)
        if ip_adapter_path is not None:
            print(f"loading image_proj_model ...")
            state_dict = torch.load(ip_adapter_path, map_location="cpu")
            image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            adapter_modules = torch.nn.ModuleList(transformer.attn_processors.values())
            adapter_modules.load_state_dict(state_dict["ip_adapter"],strict=False)


def set_lora(transformer, local_path, lora_weights=[], cond_size=512):
    checkpoint = load_checkpoint(local_path)
    update_model_with_lora(checkpoint, lora_weights, transformer, cond_size)
   
def set_multi_lora(transformer, local_paths,ip_adapter_path, lora_weights=[[]], cond_size=512):
    checkpoints = [load_checkpoint(local_path) for local_path in local_paths]
    update_model_with_multi_lora(checkpoints, lora_weights, transformer, cond_size,ip_adapter_path)

def unset_lora(transformer):
    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        lora_attn_procs[name] = FluxAttnProcessor2_0()
    transformer.set_attn_processor(lora_attn_procs)
    
    

'''
unset_lora(pipe.transformer)
lora_path = "./lora.safetensors"
lora_weights = [1, 1]
set_lora(pipe.transformer, local_path=lora_path, lora_weights=lora_weights, cond_size=512)
'''