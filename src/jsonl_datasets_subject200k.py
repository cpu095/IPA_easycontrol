from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import torch

def get_random_resolution(min_size=512, max_size=1280, multiple=16):
    # 随机生成一个大小在 min_size 到 max_size 范围内的16倍数
    resolution = random.randint(min_size // multiple, max_size // multiple) * multiple
    return resolution

# 写一个函数，调整一个浮点数是16的倍数
def multiple_16(num: float):
    return int(round(num / 16) * 16)

def make_train_dataset(args, tokenizer, accelerator=None):
    if args.train_data_dir is not None:
        print("load_data")
        dataset = load_dataset('json', data_files=args.train_data_dir)

    # 6. Get the column names for input/target.
    caption_column = args.caption_column.split(",")
    target_column = args.target_column
    if args.subject_column is not None:
        subject_columns = args.subject_column.split(",")
    if args.spacial_column is not None:
        spacial_columns= args.spacial_column.split(",")
    
    size = args.cond_size
    noise_size = get_random_resolution(max_size=args.noise_size) # maybe 768 or higher
    subject_cond_train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.resize((
                multiple_16(size * img.size[0] / max(img.size)),
                multiple_16(size * img.size[1] / max(img.size))
            ), resample=Image.BILINEAR)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.Lambda(lambda img: transforms.Pad(
                padding=(
                    int((size - img.size[0]) / 2),
                    int((size - img.size[1]) / 2),
                    int((size - img.size[0]) / 2),
                    int((size - img.size[1]) / 2) 
                ),
                fill=0  # 黑色填充
            )(img)),
            transforms.RandomAffine(degrees=25, scale=(0.7, 1.2), translate=(0.2, 0.2)),  # 添加随机缩放、平移和旋转
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    cond_train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def train_transforms(image, noise_size):
        train_transforms_ = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.resize((
                    multiple_16(noise_size * img.size[0] / max(img.size)),
                    multiple_16(noise_size * img.size[1] / max(img.size))
                ), resample=Image.BILINEAR)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        transformed_image = train_transforms_(image)
        return transformed_image
    
    def load_and_transform_cond_images(images):
        transformed_images = [cond_train_transforms(image) for image in images]
        concatenated_image = torch.cat(transformed_images, dim=1)
        return concatenated_image
    
    def load_and_transform_subject_images(images):
        transformed_images = [subject_cond_train_transforms(image) for image in images] # subject，按最长边等比例缩放
        concatenated_image = torch.cat(transformed_images, dim=1)
        return concatenated_image
    
    tokenizer_clip = tokenizer[0]
    tokenizer_t5 = tokenizer[1]

    def tokenize_prompt_clip_t5(examples, mode):
        if mode == "right2left":
            caption_list = examples[caption_column[0]]
        elif mode == "right2right":
            caption_list = examples[caption_column[1]]
        elif mode == "left2left":
            caption_list = examples[caption_column[0]]
        else:
            caption_list = examples[caption_column[1]]
            
        captions = []
        for caption in caption_list:
            if isinstance(caption, str):
                if random.random() < 0.1:
                    captions.append(" ")  # 将文本设为空
                else:
                    captions.append(caption)
            elif isinstance(caption, list):
                # take a random caption if there are multiple
                if random.random() < 0.1:
                    captions.append(" ")  # 将文本设为空
                else:
                    captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
                
        text_inputs = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        return text_input_ids_1, text_input_ids_2

    def preprocess_train(examples):
        # print("preprocess_train", examples)
        _examples = {}
        left_images = []
        right_images = []
        
        for image_path in examples[target_column]:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            left_part = img.crop((0, 0, width // 2, height))
            left_images.append(left_part)
            right_part = img.crop((width // 2, 0, width, height))
            right_images.append(right_part)
        
        prob = random.random()
        
        if prob < 0.30:
            mode = "right2left"
            target_images = left_images
            if args.subject_column is not None:
                subject_images = [right_images]
            if args.spacial_column is not None:
                spacial_images = [[Image.open(image_path).convert("RGB") for image_path in examples[spacial_columns[0]]]]

            if args.spacial_column is not None:
                _examples["cond_pixel_values"] = [load_and_transform_cond_images(spacial) for spacial in spacial_images]
            if args.subject_column is not None:
                _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
            _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
            _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples, mode)
            return _examples
        
        elif prob >=0.30 and prob < 0.50:
            mode = "right2right"
            target_images = right_images
            if args.subject_column is not None:
                subject_images = [right_images]
            if args.spacial_column is not None:
                spacial_images = [[Image.open(image_path).convert("RGB") for image_path in examples[spacial_columns[0]]]]

            if args.spacial_column is not None:
                _examples["cond_pixel_values"] = [load_and_transform_cond_images(spacial) for spacial in spacial_images]
            if args.subject_column is not None:
                _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
            _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
            _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples, mode)
            return _examples
        
        elif prob >= 0.50 and prob < 0.8:
            mode = "left2right"
            target_images = right_images
            if args.subject_column is not None:
                subject_images = [left_images]
            if args.spacial_column is not None:
                spacial_images = [[Image.open(image_path).convert("RGB") for image_path in examples[spacial_columns[1]]]]
                
            if args.spacial_column is not None:
                _examples["cond_pixel_values"] = [load_and_transform_cond_images(spacial) for spacial in spacial_images]
            if args.subject_column is not None:
                _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
            _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
            _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples, mode)
            return _examples
        
        else:
            mode = "left2left"
            target_images = left_images
            if args.subject_column is not None:
                subject_images = [left_images]
            if args.spacial_column is not None:
                spacial_images = [[Image.open(image_path).convert("RGB") for image_path in examples[spacial_columns[1]]]]
                
            if args.spacial_column is not None:
                _examples["cond_pixel_values"] = [load_and_transform_cond_images(spacial) for spacial in spacial_images]
            if args.subject_column is not None:
                _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
            _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
            _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples, mode)
            return _examples


    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
        
    if examples[0].get("subject_pixel_values") is not None: 
        subject_pixel_values = torch.stack([example["subject_pixel_values"] for example in examples])
        subject_pixel_values = subject_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        subject_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([torch.tensor(example["token_ids_clip"]) for example in examples])
    token_ids_t5 = torch.stack([torch.tensor(example["token_ids_t5"]) for example in examples])

    return {
        "cond_pixel_values": cond_pixel_values,
        "subject_pixel_values": subject_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
    }