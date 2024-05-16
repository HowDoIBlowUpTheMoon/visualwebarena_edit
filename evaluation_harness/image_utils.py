from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
import io
import requests
from PIL import Image


def get_captioning_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    if "blip2" in model_name and device != "API":
        captioning_processor = Blip2Processor.from_pretrained(model_name)
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
    elif "blip2" not in model_name: 
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )
    elif device == "API":
        print("check!")
    #     raise NotImplementedError(
    #         "API requires special handling"
    #     )

    if device != "API":
        captioning_model.to(device)

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        if device != "API":
            if prompt is None:
                # Perform VQA
                inputs = captioning_processor(
                    images=images, return_tensors="pt"
                ).to(device, dtype)
                generated_ids = captioning_model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                captions = captioning_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            else:
                # Regular captioning. Prompt is a list of strings, one for each image
                assert len(images) == len(
                    prompt
                ), "Number of images and prompts must match, got {} and {}".format(
                    len(images), len(prompt)
                )
                inputs = captioning_processor(
                    images=images, text=prompt, return_tensors="pt"
                ).to(device, dtype)
                generated_ids = captioning_model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                captions = captioning_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
        else:         
            print("Using API!")
            def pil_image_to_bytes(image, format = "JPEG"):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=format)
                img_byte_arr.seek(0)
                return img_byte_arr 
            
            files = []

            for raw_image in images:
                files.append(("images", ("test" + raw_image.format, pil_image_to_bytes(raw_image), None)))

            headers = {
                'accept': 'application/json'
            }

            url = "http://127.0.0.1:8004/caption/"

            if prompt is not None: 
                assert len(images) == len(
                    prompt
                ), "Number of images and prompts must match, got {} and {}".format(
                    len(images), len(prompt)
                )

                for ex_prompt in prompt:
                    files.append(("prompts", (None, ex_prompt)))

            # files = [
            #     ("images", ("cat.jpg", img1_bytes, "image/jpeg")), #"image/jpeg", img1_bytes)),
            #     ("images", ("dog.jpg", img2_bytes, "image/jpeg")), #"image/jpeg", img2_bytes)),
            #     ("max_new_tokens", (None, '64')),
            #     ("prompts", (None, "What is the cat laying on?")),
            #     ("prompts", (None, "What is the dog laying on?")),
            # ]

            response = requests.post(url, headers=headers, files=files)
            captions = response.json()['captions']

        return captions

    return caption_images


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
