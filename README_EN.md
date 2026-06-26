[🇨🇳 中文](./README.md) | [🇬🇧 English](./README_EN.md)
# afarTools - ComfyUI Comprehensive Plugin Pack

> **Current Version**: v3.3 
- ** Updated support for krea2 by adding -ConditioningKrea2Rebalance- to adjust parameters. Thanks to the author: https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance

`afarTools` is a self-developed comprehensive plugin pack specifically designed for ComfyUI. Through highly integrated core nodes, it greatly simplifies complex workflows from model loading, parameter configuration, to image editing, dedicated to providing users with a one-stop, high-efficiency image generation and editing experience.
<p align="center">
  <img src="./priview.jpg" width="600" alt="screenshot">
</p>

---

## 🧩 Node Introduction

### 🌟 Main Nodes
| Node Name | Function Description |
| --- | --- |
| **Unite_Ksampler** | **Comprehensive Core Node**. An all-in-one integration of model loading, parameter configuration, sampling control, reference image editing, LoRA management, and other workflow functions, enabling single-node completion of the entire image generation and editing workflow. |
| **Unite_Loader** | **Decoupled version of Unite_Ksampler**. Separates the Ksampler sampling logic to provide more flexible modular combinations. |
| **Until_Loras_Stack** | **LoRA Stack Loader**. A LoRA management node specifically designed to work with `Unite_Ksampler` / `Unite_Loader`. |

### 🛠️ Other Utility Nodes
| Node Name | Function Description |
| --- | --- |
| **CropByMask_Resize** | Crop and Resize. Supports manual bounding box selection and automatic Mask recognition for cropping. |
| **CropByMask_Restore** | The paste-back node for `CropByMask_Resize`. Supports setting an overlay model to perfectly restore the processed image. |
| **CropByMask_Resize_sam3** | Upgraded version of `CropByMask_Resize`. Incorporates SAM3 semantic recognition for more precise automatic cropping. |
| **ImageMask_Pad_Resize** | A comprehensive processing node for image Padding or Resizing. |
| **Fill_Mask_Holes** | A simple Mask filling node used to repair holes in Masks. |
| **Image_Mask_Preview** | Image and Mask overlay preview node. Supports simple processing like Mask blurring and expansion with real-time preview output. |

---

## 🚀 Compatible Models

Compatible with the following mainstream model series:

- **🤖 Qwen Series**: Qwen Image 2512, Qwen Image Edit 2509/2511, FireRed Edit
- **🌊 Flux Series**: Flux1.dev, Chroma, Flux2, Flux2 Klein, Ernie
- **🎨 Traditional SD Series**: SD1.5, SDXL, Illustrious, Noodai, Pony
- **⚡ Zimage Series**: Zimage Base, Zimage Turbo
- **📦 AIO Integrated Large Models**: Standard Checkpoint models, packaged AIO full models

---

## 💡 Core Highlight Features

### 1. Dual Large Model Loading System
- **Independent Start/Stop**: Controls independent loading and running of dual models via the `use model2` switch.
- **Dual Sampling**: Supports the Advance Double dual-model dual-sampling pipeline of the same type.
- **Smart Override**: Clip/VAE override logic; when the large model comes with its own clip/vae, it prioritizes using the model's own components.
- **Independent LoRA Binding**: lora1 binds to model 1, lora2 binds to model 2, precisely targeting and accelerating LoRAs. (*More LoRA extensions can be connected via the external `loras stack` node*)

### 2. Reference Image Adaptation
- **Dynamic Expansion**: The reference image size matching switch is hidden by default and automatically expands when a reference image is passed.
- **Auto Alignment**: When enabled, the Latent size automatically aligns with Reference Image 1.
- **Free Definition**: When disabled, you can freely customize the output resolution.

### 3. Preset System
- **Full Parameter Saving**: All parameters such as models, CLIP, VAE, ControlNet, steps, samplers, etc., can be saved as presets and loaded with one click.
- **Prompt-Exclusive Presets**: A neat trick 🔑 — by naming files with the `_p` or `_prompt` prefix, only the positive and negative prompts will be overwritten when loaded.
- **Quick Management**: Built-in preset folder open button for quick modification and deletion of preset templates.

### 4. Seed Control
- Replaces the official native seed function with more intuitive operations.
- **State Switching**: Locked state by default; turning it on switches to random mode.
- **Convenient Operations**: Supports manual input, one-click refresh, and one-click rollback to the previous seed parameters.

### 5. ControlNet / Edit Dual-Mode Mutual Exclusion Mechanism
- **Mode Mutual Exclusion**: Choose either Image Generation CN mode or Image Editing mode (most editing models have built-in reference capabilities, no need to stack CN).
- **Broad Support**: Currently supports standard CN and Model Patch (Zimage) models.
- **Input Logic**:
  - **Edit Mode**: `image1` - `image4` are all normal sequential reference images.
  - **ControlNet Mode**: When enabled, `image2` is the reference image.
  - **Zimage Type**: `image2` is the reference image, `image3` is the Inpaint reference image.

### 6. Format Support
- Supports mainstream model formats such as `gguf`, `safetensors`, `pt`, `ckpt`, etc.

---

## 📥 Installation

### Method 1: Git Clone Installation (Recommended)
Open your terminal (Terminal / CMD), navigate to the `custom_nodes` directory of ComfyUI, and execute the following commands:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-afarTools.git
```
*Restart ComfyUI to use it normally.*

### Method 2: Manual Installation
1. Click **Code** -> **Download ZIP** on the GitHub repository page to download the project zip file.
2. Unzip the file and rename the extracted folder to `ComfyUI-afarTools` (or keep the original name).
3. Place this folder into the `custom_nodes` folder under the ComfyUI root directory.
4. Restart ComfyUI.

### Note: Related dependencies
1. The plugin supports retrieving files in gguf format and depends on the pig file in the gguf plugin, so you need to have or install the gguf plugin first: https://github.com/calcuis/gguf. Otherwise, GGUF support will expire, but this does not affect the continued use of plugins.
2. In CropByMask_Resize, CropByMask_Resize_sam3 nodes will use the opencv-python library. Most plugins are actually pre-installed, so if you don't have one, you can install it yourself. Enter your own Python environment and pip install opencv-python. Actually, nodes can still use it without installation, but the efficiency and masking results are much less than support from the opencv-python package. If not installed, it is recommended to install it.

---

## 📝 Changelog

- **v3.3** (Current Version): Released the afarTools comprehensive plugin pack, including core features such as the Unite_Ksampler core node, dual large model loading system, SAM3 semantic recognition cropping, and full-parameter preset system.

---

## 🤝 Support and Feedback

If you encounter any issues during use or have better feature suggestions, feel free to submit an **Issue** or **Pull Request** on GitHub!