import time
import torch
from PIL import Image
from src.pipeline_image_emb import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from safetensors.torch import save_file
from src.all_infer_layer import set_lora, unset_lora,set_multi_lora
from tqdm import tqdm
import os
import json
from diffusers.utils import load_image
from diffusers import FluxPriorReduxPipeline
from safetensors.torch import load_file
from transformers import AutoProcessor, SiglipVisionModel


torch.cuda.set_device(0)

#pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/FLUX-redux", torch_dtype=torch.float32).to("cuda")
# checkpoint_path = "/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/models/canny_model_0403/checkpoint-15000"

# # 加载 image_embedder 和 image_encoder 权重
# image_embedder_state_dict = load_file(os.path.join(checkpoint_path, "image_embedder.safetensors"))
# image_encoder_state_dict = load_file(os.path.join(checkpoint_path, "image_encoder.safetensors"))

# # 将加载的权重分别赋值给模型
# pipe_prior_redux.image_embedder.load_state_dict(image_embedder_state_dict)
# pipe_prior_redux.image_encoder.load_state_dict(image_encoder_state_dict)

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=128):
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


class ImageProcessor:
    def __init__(self, path, image_encoder_path):
        self.device = "cuda"
        self.pipe = FluxPipeline.from_pretrained(path, torch_dtype=torch.bfloat16, device=self.device)
        transformer = FluxTransformer2DModel.from_pretrained(path, subfolder="transformer",torch_dtype=torch.bfloat16, device=self.device)
        ##
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(device=self.device, dtype=torch.bfloat16)
        self.clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)
        self.image_proj_model = self.init_proj(device=self.device)
        
        ##
        self.pipe.transformer = transformer
        self.pipe.to(self.device)
    

    def init_proj(self,device):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=128,
        ).to(device=device, dtype=torch.bfloat16)
        
        return image_proj_model

    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.bfloat16)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    

    def clear_cache(self, transformer):
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()
        
    def process_image(self, oriimage_path, prompt='', subject_imgs=[], spatial_imgs=[], height=768, width=768, output_path=None, seed=42):
        if len(spatial_imgs)>0:
            spatial_ls = [Image.open(image_path).convert("RGB") for image_path in spatial_imgs]
        else:
            spatial_ls = []
        if len(subject_imgs)>0:
            subject_ls = [Image.open(image_path).convert("RGB") for image_path in subject_imgs]
        else:
            subject_ls = []
        original_width, original_height = spatial_ls[0].size
        ori_image=load_image(oriimage_path)
        image_prompt_embeds = self.get_image_embeds(
            pil_image=ori_image, clip_image_embeds=None
        )
        
        prompt = prompt
        image = self.pipe(
            prompt,
            #pil_image=ori_image,
            image_emb=image_prompt_embeds,
            height=int(height),
            width=int(width),
            guidance_scale=3.5,
            num_inference_steps=100,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed), 
            subject_images=subject_ls,
            spatial_images=spatial_ls,
            cond_size=512,
        ).images[0]
        #self.clear_cache(self.pipe.transformer)

        #self.clear_cache(self.pipe)
            # 遍历生成的图片并分别保存
        images = [image]
        for idx, img in enumerate(images):
            img_resized = img.resize((original_width, original_height))  # 调整为原始尺寸
            # 保存每张图片，使用不同的文件名防止覆盖
            img_save_path = output_path.replace(".png", f"_{idx}.png")
            img_resized.save(img_save_path)
            print(f"保存图片：{img_save_path}")
       
            
if __name__ == "__main__":
    # 示例使用
    path = "/opt/liblibai-models/model-weights/black-forest-labs/FLUX.1-dev"
    image_encoder_path = "/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/IPA_easycontrol/siglip"
    processor = ImageProcessor(path,image_encoder_path=image_encoder_path)
    # ground_truth_images_dir = '/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/blurred_datasets'
    # ori_image_path='/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/blurred_datasets'
    # output_dir = '/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/output/blurref/0412/train_datasets'
    # prompts_file = '/opt/liblibai-models/user-workspace/zhangyuxuan/datasets/concept101/prompt_all_.json'
    
    # eval_images = [file_path for file_path in os.listdir(ground_truth_images_dir) if file_path.endswith('.jpg')]
    # print(f"Total file count: {len(eval_images)}")
    # os.makedirs(output_dir, exist_ok=True)
    # with open(prompts_file, 'r') as f:
    #     prompts = json.load(f)
    
    # for image_path in tqdm(eval_images):
    #     path_parts = image_path[:-4].split('_')
    #     img_class, img_idx, prompt_idx = '_'.join(path_parts[:-2]), path_parts[-2], path_parts[-1]
    #     #prompt = prompts[img_class][int(prompt_idx)]
    #     canny_input_path = os.path.join(ground_truth_images_dir, image_path)
    #     image_input_path= os.path.join(ori_image_path, image_path)
    #     output_path = os.path.join(output_dir, image_path)
    #     tqdm.write(f"Processing image: {image_path} to {output_path}")
    #     processor.process_image(subjects=[canny_input_path], output_path=output_path,pr=[image_input_path],num=1)
    
    lora_path1="/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/models/blurred_model_0409_resume/checkpoint-8000/lora.safetensors"
    lora_path2="/opt/liblibai-models/user-workspace/zhangyuxuan/project/easycontrol/code0221/models/canny_model_0226/checkpoint-60000/lora.safetensors"
    ori_image_path="/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/resize_ori/source_98.jpg"
    set_multi_lora(processor.pipe.transformer, [lora_path1, lora_path2], lora_weights=[[1],[1]],cond_size=512,ip_adapter_path="/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/FLUX-IPAdapter/ip-adapter.bin")
    prompt='Tower Bridge'
    spatial_imgs=["/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/blurred_size_same_canny/source_98.jpg","/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/input/source_98.jpg"]
    subject_imgs=[]
    seed = 42
    processor.process_image(prompt=prompt, oriimage_path=ori_image_path,subject_imgs=subject_imgs, spatial_imgs=spatial_imgs, height=1024, width=1024, seed=seed, 
                            output_path="/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/output/canny_blurred/100.png")