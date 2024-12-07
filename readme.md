# Golden Noise Generator for ComfyUI (SDXL)

## Overview

This is a ComfyUI custom node that implements the Golden Noise Generator from the paper ["Golden Noise for Diffusion Models: A Learning Framework"](https://arxiv.org/abs/2411.09502), specifically optimized for SDXL. The node enhances the initial latent noise used in the diffusion process to improve image quality and semantic consistency.

## Features

- Optimized for SDXL models
- Transformer-based noise processing using Swin Transformer
- SVD-based noise enhancement
- Residual connections for controlled noise modification
- Seamless integration with ComfyUI workflows

## Installation

1. Navigate to your ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/Golden-Noise-for-Diffusion-Models
```

3. Install required dependencies:
```bash
pip install timm einops
```

## Usage

The node appears in ComfyUI as "Golden Noise Generator" under the "latent/noise" category.

### Node Inputs
- **Source**: Choose between CPU or GPU processing
- **Use Transformer**: Enable/disable Swin Transformer processing
- **Use SVD**: Enable/disable SVD-based noise enhancement
- **Residual**: Enable/disable residual connections
- **Seed**: Random seed for reproducibility
- **Width**: Output width (must be divisible by 8)
- **Height**: Output height (must be divisible by 8)
- **Batch Size**: Number of samples to generate

### Example Workflow
1. Load your SDXL checkpoint
2. Add the "Golden Noise Generator" node
3. Connect the noise output to your KSampler's latent input
4. Configure settings (recommended to start with all options enabled)
5. Generate!

### Recommended Settings for SDXL
- Resolution: 1024x1024 (standard SDXL resolution)
- Use both transformer and SVD processing
- Enable residual connections for stable results
- Adjust seed for different noise patterns

## Pre-trained Weights

link to the weights [here](https://drive.google.com/drive/folders/1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt)

they are pth files so be careful and i would suggest converting them to safetensors.

i provide a very simple script to convert them to safetensors.
```bash
python convert.py --input weights --output weights
```

## Citation

If you use this node in your work, please cite the original paper:

```bibtex
@misc{zhou2024goldennoisediffusionmodels,
      title={Golden Noise for Diffusion Models: A Learning Framework}, 
      author={Zikai Zhou and Shitong Shao and Lichen Bai and Zhiqiang Xu and Bo Han and Zeke Xie},
      year={2024},
      eprint={2411.09502},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Original paper by Zikai Zhou et al.
- ComfyUI community
- Swin Transformer team for the pre-trained models