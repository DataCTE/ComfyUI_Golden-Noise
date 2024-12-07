# Golden Noise Generator for ComfyUI

## Overview

This is a ComfyUI custom node implementation of the Golden Noise Generator from the paper ["Golden Noise for Diffusion Models: A Learning Framework"](https://arxiv.org/abs/2411.09502). The node transforms random Gaussian noise into "golden noise" by adding desirable perturbations derived from text prompts to boost the overall quality and semantic faithfulness of synthesized images.

## Features

- Supports SDXL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT models
- Integrates seamlessly with ComfyUI's workflow
- Includes both transformer and SVD-based noise processing
- Optional residual connections for fine-tuned control

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

After installation, you'll find a new node in ComfyUI called "Golden Noise Generator" under the "latent/noise" category.

### Node Inputs
- **Source**: Choose between CPU or GPU processing
- **Use Transformer**: Enable/disable transformer-based noise processing
- **Use SVD**: Enable/disable SVD-based noise processing
- **Residual**: Enable/disable residual connections
- **Seed**: Random seed for reproducibility
- **Width**: Output width (must be divisible by 8)
- **Height**: Output height (must be divisible by 8)
- **Batch Size**: Number of samples to generate

### Example Workflow
1. Add the "Golden Noise Generator" node to your workflow
2. Connect it to your model's latent input
3. Configure the settings as needed
4. Run your workflow with enhanced noise generation

## Pre-trained Weights

Download the pre-trained weights from [this Google Drive link](https://drive.google.com/drive/folders/1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt?usp=drive_link) and place them in the appropriate model directory.

## Citation

If you use this node in your work, please cite the original paper:

```bibtex
@misc{zhou2024goldennoisediffusionmodels,
      title={Golden Noise for Diffusion Models: A Learning Framework}, 
      author={Zikai Zhou and Shitong Shao and Lichen Bai and Zhiqiang Xu and Bo Han and Zeke Xie},
      year={2024},
      eprint={2411.09502},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.09502}, 
}
```

## License

This project is licensed under the same terms as the original Golden Noise for Diffusion Models implementation.

## Acknowledgments

- Original implementation by Zikai Zhou et al.
- ComfyUI community for the framework and support