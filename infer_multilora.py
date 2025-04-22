import time
import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from safetensors.torch import save_file
from src.lora_helper import set_lora, unset_lora,set_multi_lora
from tqdm import tqdm
import os
import json
from diffusers.utils import load_image
from diffusers import FluxPriorReduxPipeline
from safetensors.torch import load_file
torch.cuda.set_device(1)

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/FLUX-redux", torch_dtype=torch.bfloat16).to("cuda")
checkpoint_path = "/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/models/canny_model_0403/checkpoint-15000"

# # 加载 image_embedder 和 image_encoder 权重
image_embedder_state_dict = load_file(os.path.join(checkpoint_path, "image_embedder.safetensors"))
image_encoder_state_dict = load_file(os.path.join(checkpoint_path, "image_encoder.safetensors"))

# # 将加载的权重分别赋值给模型
pipe_prior_redux.image_embedder.load_state_dict(image_embedder_state_dict)
pipe_prior_redux.image_encoder.load_state_dict(image_encoder_state_dict)


class ImageProcessor:
    def __init__(self, path):
        device = "cuda"
        self.pipe = FluxPipeline.from_pretrained(path, torch_dtype=torch.bfloat16, device=device)
        transformer = FluxTransformer2DModel.from_pretrained(path, subfolder="transformer",torch_dtype=torch.bfloat16, device=device)
        self.pipe.transformer = transformer
        self.pipe.to(device)
        
    def clear_cache(self, transformer):
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()
        
    def process_image(self, prompt='', subject_imgs=[], spatial_imgs=[], height = 768, width = 768, output_path=None, seed=42):
        if len(spatial_imgs)>0:
            spatial_ls = [Image.open(image_path).convert("RGB") for image_path in spatial_imgs]
        else:
            spatial_ls = []
        if len(subject_imgs)>0:
            subject_ls = [Image.open(image_path).convert("RGB") for image_path in subject_imgs]
        else:
            subject_ls = []
        original_width, original_height = spatial_ls[0].size
        image=load_image("/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/resize_ori/source_98.jpg")
        pipe_prior_output = pipe_prior_redux(image)

        prompt = prompt
        image = self.pipe(
            #prompt,
            height=int(height),
            width=int(width),
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed), 
            subject_images=subject_ls,
            spatial_images=spatial_ls,
            cond_size=512,
            **pipe_prior_output
        ).images[0]
        self.clear_cache(self.pipe.transformer)

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
    processor = ImageProcessor(path)
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
    set_multi_lora(processor.pipe.transformer, [lora_path1, lora_path2], lora_weights=[[1.2],[1]],cond_size=512)
    prompt='Tower Bridge'
    spatial_imgs=["/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/blurred_size_same_canny/source_98.jpg","/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/input/source_98.jpg"]
    subject_imgs=[]
    seed = 42
    processor.process_image(prompt=prompt, subject_imgs=subject_imgs, spatial_imgs=spatial_imgs, height=1024, width=1024, seed=seed, 
                            output_path="/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/output/canny_blurred_redux/98.png")