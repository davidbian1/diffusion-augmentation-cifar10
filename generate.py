from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from huggingface_hub import hf_hub_download
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

class DDPMPipelinenew(DiffusionPipeline):
    def __init__(self, unet, scheduler, num_classes: int):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.num_classes = num_classes
        self._device = unet.device  # Ensure the pipeline knows the device

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 64,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        # Ensure class_labels is on the same device as the model
        class_labels = class_labels.to(self._device)
        if class_labels.ndim == 0:
            class_labels = class_labels.unsqueeze(0).expand(batch_size)
        else:
            class_labels = class_labels.expand(batch_size)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self._device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Ensure the class labels are correctly broadcast to match the input tensor shape
            model_output = self.unet(image, t, class_labels).sample

            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def to(self, device: torch.device):
        self._device = device
        self.unet.to(device)
        return self

def load_pipeline(repo_id, device, num_classes=10):
    unet = UNet2DModel.from_pretrained(repo_id, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
    pipeline = DDPMPipelinenew(unet=unet, scheduler=scheduler, num_classes=num_classes)
    return pipeline.to(device)  # Move the entire pipeline to the device

def save_images_locally(images, save_dir, epoch, class_label):
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(images):
        image_path = os.path.join(save_dir, f"image_epoch{epoch}_class{class_label}_idx{i}.png")
        image.save(image_path)

def generate_images(pipeline, class_label, save_dir, target_fraction=0.1, fraction=0.1, batch_size=64,  num_inference_steps=1000, epoch = 0):
    to_generate = int((target_fraction - fraction) * 5000)
    all_images = []
    generator = torch.Generator(device=pipeline._device).manual_seed(0)

    while to_generate > 0:
        current_batch_size = min(batch_size, to_generate)

        class_labels = torch.tensor([class_label] * current_batch_size).to(pipeline._device)
        images = pipeline(
            generator=generator,
            batch_size=current_batch_size,
            num_inference_steps=num_inference_steps,
            class_labels=class_labels,
            output_type="pil",
        ).images
        save_images_locally(images, save_dir, epoch, class_label)
        all_images.extend(images)
        to_generate -= current_batch_size

    return all_images

def create_image_grid(images, save_path, grid_size=8): 
    assert len(images) == grid_size ** 2, "Number of images must be equal to grid_size squared"
    width, height = images[0].size
    grid_img = Image.new('RGB', (grid_size * width, grid_size * height))
    
    for i, image in enumerate(images):
        x = i % grid_size * width
        y = i // grid_size * height
        grid_img.paste(image, (x, y))
    
    grid_img.save(save_path)
    return grid_img