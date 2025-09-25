import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
)
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from ip_adapter import IPAdapterPlusXL
from model.dit import DiT_Llama
from model.pipeline_pit import PiTPipeline
from training.train_config import TrainConfig
from utils import vis_utils, bezier_utils
from itertools import combinations

import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
#from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from dds_cloudapi_sdk.image_resizer import image_to_base64

API_TOKEN = ""
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

BOX_THRESHOLD = 0.2
IOU_THRESHOLD = 0.8
WITH_SLICE_INFERENCE = False
SLICE_WH = (480, 480)
OVERLAP_RATIO = (0.2, 0.2)
GROUNDING_MODEL ="GroundingDino-1.6-Pro"
SAM2_CHECKPOINT = "/data/data/shibu/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available()  else "cpu"
OUTPUT_DIR = Path("inference/output_3")

# Step 1: initialize the config
token = API_TOKEN
config = Config(token)

# Step 2: initialize the client
client = Client(config)

JSON_FILE = "/data/data/shibu/PiT/scripts/object-image-pairs.json"

with open(JSON_FILE, "r") as f:
    data = json.load(f)


# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
# model_id = GROUNDING_MODEL
# processor = AutoProcessor.from_pretrained(model_id)
# grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)





def paste_on_background(image, background, min_scale=0.4, max_scale=0.8, scale=None):
    # Calculate aspect ratio and determine resizing based on the smaller dimension of the background
    aspect_ratio = image.width / image.height
    scale = random.uniform(min_scale, max_scale) if scale is None else scale
    new_width = int(min(background.width, background.height * aspect_ratio) * scale)
    new_height = int(new_width / aspect_ratio)

    # Resize image and calculate position
    image = image.resize((new_width, new_height), resample=Image.LANCZOS)
    pos_x = random.randint(0, background.width - new_width)
    pos_y = random.randint(0, background.height - new_height)

    # Paste the image using its alpha channel as mask if present
    background.paste(image, (pos_x, pos_y), image if "A" in image.mode else None)
    return background


def set_seed(seed: int):
    """Ensures reproducibility across multiple libraries."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random seed
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking to avoid randomness


# Inside main():


@dataclass
class RunConfig:
    prior_path: Path
    crops_dir: Path
    output_dir: Path
    prior_repo: Optional[str] = None
    prior_guidance_scale: float = 1.0
    drop_cond: bool = True
    n_randoms: int = 20
    as_sketch: bool = False
    scale: float = 2.0
    use_empty_ref: bool = True


@pyrallis.wrap()
def main(cfg: RunConfig):
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prior_cfg_path = "/data/data/shibu/PiT/configs/cfg.yaml"
    # Load model_cfg from file
    model_cfg: TrainConfig = pyrallis.load(TrainConfig, open(prior_cfg_path, "r"))

    prior_ckpt_path = "/data/data/shibu/PiT/training_results/train_animals/prior_250000.ckpt"

    weight_dtype = torch.float32
    device = "cuda"
    prior = DiT_Llama(
        embedding_dim=2048,
        hidden_dim=model_cfg.hidden_dim,
        n_layers=model_cfg.num_layers,
        n_heads=model_cfg.num_attention_heads,
    )

    print(prior_ckpt_path)
    print(output_dir)
    print(cfg.crops_dir)
    prior.load_state_dict(torch.load(prior_ckpt_path))

    image_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    ip_ckpt_path = hf_hub_download(
        repo_id="h94/IP-Adapter",
        filename="ip-adapter-plus_sdxl_vit-h.bin",
        subfolder="sdxl_models",
        local_dir="pretrained_models",
    )

    ip_model = IPAdapterPlusXL(
        image_pipe,
        "models/image_encoder",
        ip_ckpt_path,
        device,
        num_tokens=16,
    )

    image_processor = ip_model.clip_image_processor

    empty_image = Image.new("RGB", (256, 256), (255, 255, 255))
    zero_image = torch.Tensor(image_processor(empty_image)["pixel_values"][0])
    zero_image_embeds = ip_model.get_image_embeds(zero_image.unsqueeze(0), skip_uncond=True)

    prior_pipeline = PiTPipeline(
        prior=prior,
    )
    prior_pipeline = prior_pipeline.to(device)

    set_seed(42)
    
    
    
    print("Start segmentation...")

    # === Iterate over all entries ===
    for entry_idx, entry in enumerate(data):
        img_path = entry["image"]
        parts = entry["parts"]

        try:
            image_b64 = image_to_base64(img_path)
        except Exception as e:
            print(f"[{entry_idx}] Skipping image: {img_path} (error: {e})")
            continue

        for part_idx, text_prompt in enumerate(parts):

            image_basename = os.path.splitext(os.path.basename(img_path))[0]
            prompt_clean = text_prompt.strip().replace(" ", "_").replace(".", "")
            filename = f"{entry_idx:02d}_{part_idx:02d}_{image_basename}_{prompt_clean}.jpg"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(save_path):
                print(f"[{entry_idx}] Skipping existing output: {save_path}")
                continue

            print(f"Processing: {img_path} | prompt: '{text_prompt.strip()}'")

            try:
                task = V2Task(
                    api_path="/v2/task/grounding_dino/detection",
                    api_body={
                        "model": GROUNDING_MODEL,
                        "image": image_b64,
                        "prompt": {
                            "type": "text",
                            "text": text_prompt
                        },
                        "targets": ["bbox"],
                        "bbox_threshold": BOX_THRESHOLD,
                        "iou_threshold": IOU_THRESHOLD,
                    }
                )

                client.run_task(task)
                result = task.result
                objects = result["objects"]

                input_boxes = []
                confidences = []
                class_names = []
                class_ids = []

                for idx, obj in enumerate(objects):
                    input_boxes.append(obj["bbox"])
                    confidences.append(obj["score"])
                    cls_name = obj["category"].lower().strip()
                    class_names.append(cls_name)

                input_boxes = np.array(input_boxes)
                class_ids = np.array(class_ids)

                if len(confidences) == 0:
                    print(f" No detection found for part: {text_prompt}")
                    continue

                top_idx = np.argmax(confidences)
                top_box = input_boxes[top_idx]
                top_box_array = np.expand_dims(top_box, axis=0)

                # === SAM2 mask prediction ===
                image = Image.open(img_path)
                sam2_predictor.set_image(np.array(image.convert("RGB")))

                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=top_box_array,
                    multimask_output=False,
                )

                img = cv2.imread(img_path)
                mask = masks.squeeze()
                white_bg = np.ones_like(img, dtype=np.uint8) * 255
                for c in range(3):
                    white_bg[:, :, c] = np.where(mask, img[:, :, c], white_bg[:, :, c])

                ys, xs = np.where(mask)
                if ys.size > 0 and xs.size > 0:
                    min_y, max_y = ys.min(), ys.max()
                    min_x, max_x = xs.min(), xs.max()
                    cropped_white_bg = white_bg[min_y:max_y + 1, min_x:max_x + 1]
                else:
                    cropped_white_bg = white_bg

                cv2.imwrite(save_path, cropped_white_bg)
                print(f" Saved to {save_path}")

            except Exception as e:
                print(f"Error processing {img_path} with prompt '{text_prompt}': {e}")
                continue

    print("Segmentation finished.")
    
    
    # #Read all crops from the dir
    # crop_sets = []
    # unordered_crops = []
    # for crop_dir_path in cfg.crops_dir.iterdir():
    #     unordered_crops.append(crop_dir_path)
    # unordered_crops = sorted(unordered_crops, key=lambda x: x.stem)

    # if len(unordered_crops) > 0:
    #     for _ in range(cfg.n_randoms):
    #         n_crops = random.randint(1, min(2, len(unordered_crops)))
    #         crop_paths = random.sample(unordered_crops, n_crops)
    #         # Some of the paths might be dirs, if it is a dir, take a random file from it
    #         crop_paths = [c if c.is_file() else random.choice([f for f in c.iterdir()]) for c in crop_paths]
    #         crop_sets.append(crop_paths)

    crop_base_dir = Path("/data/data/shibu/PiT/inference/output_3")
    all_crop_paths = []
    
    # Collect a representative file from each subfolder or directly add files
    for item in sorted(crop_base_dir.iterdir()):
        if item.is_dir():
            files = list(item.glob("*"))
            if files:
                all_crop_paths.append(random.choice(files))  # pick one from folder
        elif item.is_file():
            all_crop_paths.append(item)

    # Generate all possible 2-combinations (without repetition)
    crop_sets = list(combinations(all_crop_paths, 2))
        
    if model_cfg.use_ref:
        if cfg.use_empty_ref:
            print(f"----- USING EMPTY GRIDS -----")
            augmented_crop_sets = [[None] + list(crop_set) for crop_set in crop_sets]
        else:
            print(f"----- USING REFERENCE GRIDS -----")
            augmented_crop_sets = []
            refs_dir = Path("assets/ref_grids")
            refs = [f for f in refs_dir.iterdir()]
            for crop_set in crop_sets:
                # Choose a subset of refs
                chosen_refs = random.sample(refs, 1)  # [None]  # + random.sample(refs, 5)
                for ref in chosen_refs:
                    augmented_crop_sets.append([ref] + crop_set)

        crop_sets = augmented_crop_sets

    random.shuffle(crop_sets)
    

    for crop_paths in tqdm(crop_sets):
        out_name = f"{random.randint(0, 1000000)}"

        processed_crops = []
        input_images = []
        captions = []

        # Extend to >3 with Nones
        while len(crop_paths) < 3:
            crop_paths.append(None)

        for path_ind, path in enumerate(crop_paths):
            if path is None:
                image = Image.new("RGB", (224, 224), (255, 255, 255))
            else:
                image = Image.open(path).convert("RGB")
                if path_ind > 0 or not model_cfg.use_ref:
                    background = Image.new("RGB", (1024, 1024), (255, 255, 255))
                    image = paste_on_background(image, background, scale=0.92)
                else:
                    image = image.resize((1024, 1024))
                if cfg.as_sketch and random.random() < 0.5:
                    num_lines = random.randint(8, 15)
                    image = bezier_utils.get_sketch(image, total_curves=num_lines, drop_line_prob=0.1)
                input_images.append(image)
                # Name should be parent directory name
                captions.append(path.parent.stem)
            processed_image = (
                torch.Tensor(image_processor(image)["pixel_values"][0]).to(device).unsqueeze(0).to(weight_dtype)
            )
            processed_crops.append(processed_image)
            print(len(processed_crops))

        image_embed_inputs = []
        for crop_ind in range(len(processed_crops)):
            image_embed_inputs.append(ip_model.get_image_embeds(processed_crops[crop_ind], skip_uncond=True))
        crops_input_sequence = torch.cat(image_embed_inputs, dim=1)

        for _ in range(4):
            seed = random.randint(0, 1000000)
            for scale in [cfg.scale]:
                negative_cond_sequence = torch.zeros_like(crops_input_sequence)
                embeds_len = zero_image_embeds.shape[1]
                for i in range(0, negative_cond_sequence.shape[1], embeds_len):
                    negative_cond_sequence[:, i : i + embeds_len] = zero_image_embeds.detach()

                img_emb = prior_pipeline(
                    cond_sequence=crops_input_sequence,
                    negative_cond_sequence=negative_cond_sequence,
                    num_inference_steps=25,
                    num_images_per_prompt=1,
                    guidance_scale=scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).image_embeds

                for seed_2 in range(1):
                    images = ip_model.generate(
                        image_prompt_embeds=img_emb,
                        num_samples=1,
                        num_inference_steps=50,
                    )
                    input_images += images
                    captions.append(f"prior_s {seed}, cfg {scale}")  # , unet_s {seed_2}")
        # The rest of the results will just be in the dir
        gen_images = vis_utils.create_table_plot(images=input_images, captions=captions)

        gen_images.save(output_dir / f"{out_name}.jpg")

        # Also save the divided images in a separate folder whose name is the same as the output image
        divided_images_dir = output_dir / f"{out_name}_divided"
        divided_images_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(input_images):
            img.save(divided_images_dir / f"{i}.jpg")
    print("Done!")



if __name__ == "__main__":
    main()
