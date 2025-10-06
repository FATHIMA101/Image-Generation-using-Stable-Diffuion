import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline, AutoProcessor, AutoModelForUniversalSegmentation, AutoConfig
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
import os

                                  #### OBJECT REPLACER CLASS ####

class ObjectReplacer:
    def __init__(self):
        # Load OneFormer for better segmentation
        model_name = "shi-labs/oneformer_ade20k_swin_tiny"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.seg_model = AutoModelForUniversalSegmentation.from_pretrained(model_name)

        # Get ADE20K classes from config
        config = AutoConfig.from_pretrained(model_name)
        self.class_names = list(config.id2label.values())
        print(f"Loaded {len(self.class_names)} object classes for detection")

        # Load the SD inpainting model with proper caching
        model_id = "runwayml/stable-diffusion-inpainting"
        try:
            # First try loading normally
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                resume_download=True
            )
        except Exception as e:
            print(f"Regular loading failed, trying manual download: {e}")
            # If that fails, try manual download
            cache_dir = snapshot_download(
                model_id,
                resume_download=True,
                local_files_only=False
            )
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                cache_dir,
                torch_dtype=torch.float16
            )

        # Move to GPU if available
        if torch.cuda.is_available():
            self.seg_model = self.seg_model.to("cuda")
            self.pipe = self.pipe.to("cuda")
            print("Models loaded on GPU")
        else:
            print("GPU not available, using CPU (this will be slow)")

                                 ###### Generate Mask Function ######

    def generate_mask(self, image: Image, target_object: str) -> Image:
        """
        Generate a more accurate mask using OneFormer.
        """
        # Prepare image for segmentation
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        # Get segmentation
        outputs = self.seg_model(**inputs)
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]

        # Convert to numpy
        seg_map = seg_map.cpu().numpy()

        # Create mask based on class names
        mask = np.zeros_like(seg_map, dtype=np.uint8)

        # Find matching classes
        target_classes = [i for i, name in enumerate(self.class_names)
                         if target_object.lower() in name.lower()]

        if not target_classes:
            print(f"Warning: '{target_object}' not found in available classes.")
            print("Available classes similar to your request:")
            # Show similar classes to help user
            import difflib
            similar_classes = difflib.get_close_matches(target_object.lower(),
                                                      [name.lower() for name in self.class_names],
                                                      n=5,
                                                      cutoff=0.5)
            for cls in similar_classes:
                print(f"- {cls}")
            return None

        # Create mask for matching classes
        for class_idx in target_classes:
            mask[seg_map == class_idx] = 255

        # Dilate mask slightly to ensure coverage
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask, structure=np.ones((5,5))).astype(np.uint8) * 255

        mask_coverage = (mask > 0).sum() / mask.size * 100
        if mask_coverage < 1:
            print(f"Warning: Very small mask coverage ({mask_coverage:.1f}%). The object might not be detected correctly.")

        return Image.fromarray(mask)


      ########## Replace Object Function ##########

    def replace_object(self,
                      image: Image,
                      mask: Image,
                      prompt: str,
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5) -> Image:
        """
        Enhanced replacement function with better prompt handling.
        """
        # Resize images to be multiples of 8 (SD requirement)
        width, height = image.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        image = image.resize((new_width, new_height))
        mask = mask.resize((new_width, new_height))

        # Ensure mask is binary
        mask = mask.convert('L')

        # Enhanced prompt for better results
        full_prompt = f"high quality, detailed, {prompt}"
        negative_prompt = "low quality, blurry, distorted, deformed"

        print(f"Generating with prompt: {full_prompt}")

        # Generate the replacement
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        return result

        ########## Process Image Function #########

    def process_image(self,
                     image_path: str,
                     target_object: str,
                     replacement_prompt: str,
                     num_inference_steps: int = 50) -> tuple:
        """
        Complete pipeline with enhanced processing.
        """
        # Load and preprocess image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Generate mask
        print(f"Generating mask for {target_object}...")
        mask = self.generate_mask(image, target_object)

        if mask is None:
            raise ValueError(f"Could not generate mask for '{target_object}'. Please check the suggested similar classes above.")

        # Replace object
        print(f"Replacing with '{replacement_prompt}'...")
        result = self.replace_object(
            image,
            mask,
            replacement_prompt,
            num_inference_steps=num_inference_steps
        )

        return image, mask, result


        ########## Display Results Function ########

    def display_results(self, original: Image, mask: Image, result: Image):
        """Display the original image, mask, and result side by side."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(original)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Generated Mask")
        ax2.axis("off")

        ax3.imshow(result)
        ax3.set_title("Result")
        ax3.axis("off")

        plt.tight_layout()
        plt.show()

        ######## demonstration ##########


if __name__ == "__main__":
    replacer = ObjectReplacer()

    # Replace object in image
    try:
        image, mask, result = replacer.process_image(
            image_path="/content/yellow car.jpg",
            target_object="car",  # Object to replace
            replacement_prompt="black ferrari sports car",  # What to replace it with
            num_inference_steps=75  # Increased for better quality
        )

        # Display results
        replacer.display_results(image, mask, result)
    except Exception as e:
        print(f"Error occurred: {e}")
