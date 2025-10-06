# Object Replacer with OneFormer and Stable Diffusion

An AI-powered image editing tool that automatically detects and replaces specific objects in an image using semantic segmentation (OneFormer) and inpainting (Stable Diffusion). Users can interactively define the object to be replaced and provide a natural language prompt for what it should become enabling photorealistic, high-quality transformations.

# Features

 Accurate Object Segmentation using OneFormer
 for identifying specific elements (e.g., cars, people, animals).

 Prompt-Based Object Replacement using Stable Diffusion Inpainting.

 Semantic Matching with support for fuzzy class matching if the exact class is not found.

 Mask Visualization alongside original and final images.

 Customizable Parameters such as inference steps and guidance scale.

 GPU Support for faster processing (falls back to CPU if not available).

# How It Works

Image Input: Load an image using its file path.

Object Detection: The model segments the image and detects the specified object class.

Mask Creation: A binary mask is created around the object.

Prompt-Based Replacement: Stable Diffusion inpaints the masked region based on your descriptive prompt.

Result Display: Original image, generated mask, and final result are shown side-by-side.

# Output
<img width="1417" height="511" alt="output_0" src="https://github.com/user-attachments/assets/d2e0160b-01f2-4c39-85fa-3a962ce4e355" />
<img width="1427" height="497" alt="output 3" src="https://github.com/user-attachments/assets/e8171cea-2e49-416b-9b07-b626167da0e0" />


