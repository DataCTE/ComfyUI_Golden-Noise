import torch
import torch.nn as nn
import einops
from torch.nn import functional as F
from timm import create_model
from torch.jit import Final
from timm.layers import use_fused_attn
import comfy.model_management

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

MAX_RESOLUTION = 8192

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NoiseTransformer(nn.Module):
    def __init__(self, resolution=128):
        super().__init__()
        self.target_size = 224  # Swin transformer's expected input size
        self.resolution = resolution
        # Remove fixed size from upsample/downsample
        self.upconv = nn.Conv2d(7,4,(1,1),(1,1),(0,0))
        self.downconv = nn.Conv2d(4,3,(1,1),(1,1),(0,0))
        self.swin = create_model("swin_tiny_patch4_window7_224",pretrained=True)

    def forward(self, x, residual=False):
        # Get input dimensions
        b, c, h, w = x.shape
        
        # Scale to Swin's required input size while maintaining aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = self.target_size
            new_h = int(self.target_size / aspect_ratio)
        else:
            new_h = self.target_size
            new_w = int(self.target_size * aspect_ratio)
        
        # Ensure dimensions are multiples of 32 (Swin requirement)
        new_h = ((new_h + 31) // 32) * 32
        new_w = ((new_w + 31) // 32) * 32
        
        # Dynamic up/downsampling
        x_up = F.interpolate(x, size=(new_h, new_w), mode='bilinear')
        x_down = self.downconv(x_up)
        features = self.swin.forward_features(x_down)
        x_processed = self.downsample(features)
        output = self.upconv(x_processed)
        
        # Resize back to input dimensions
        output = F.interpolate(output, size=(h, w), mode='bilinear')
        
        if residual:
            output = output + x
            
        return output

class SVDNoiseUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=128):
        super(SVDNoiseUnet, self).__init__()
        
        # Calculate input/output sizes based on resolution
        self.resolution = resolution
        _in = resolution * in_channels // 2
        _out = resolution * out_channels // 2
        
        self.mlp1 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(_in, _out),
        )
        self.attention = Attention(_out)
        self.bn = nn.BatchNorm2d(_out)
        self.mlp4 = nn.Sequential(
            nn.Linear(_out, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _out),
        )

    def forward(self, x, residual=False):
        b, c, h, w = x.shape
        original_x = x
        
        # Ensure input is reshaped to match the expected resolution
        if h != self.resolution or w != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear')
        
        x = einops.rearrange(x, "b (a c) h w -> b (a h) (c w)", a=2, c=2)
        U, s, V = torch.linalg.svd(x)
        U_T = U.permute(0, 2, 1)
        out = self.mlp1(U_T) + self.mlp2(V) + self.mlp3(s).unsqueeze(1)
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s
        pred = U @ torch.diag_embed(out) @ V
        result = einops.rearrange(pred, "b (a h) (c w) -> b (a c) h w", a=2, c=2)
        
        # Resize back to original dimensions if needed
        if h != self.resolution or w != self.resolution:
            result = F.interpolate(result, size=(h, w), mode='bilinear')
            
        if residual:
            result = result + original_x
        return result

class GoldenNoiseLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source": (["CPU", "GPU"],),
            "use_transformer": ("BOOLEAN", {"default": True}),
            "use_svd": ("BOOLEAN", {"default": True}),
            "residual": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_noise"
    CATEGORY = "latent/noise"

    def __init__(self):
        self.noise_transformer = None
        self.svd_noise_unet = None

    def initialize_models(self, device):
        if self.noise_transformer is None:
            self.noise_transformer = NoiseTransformer(resolution=128).to(device)
            self.noise_transformer.eval()
        
        if self.svd_noise_unet is None:
            self.svd_noise_unet = SVDNoiseUnet(in_channels=4, out_channels=4, resolution=128).to(device)
            self.svd_noise_unet.eval()

    def generate_noise(self, source, use_transformer, use_svd, residual, seed, width, height, batch_size):
        # Set device based on source
        if source == "CPU":
            device = "cpu"
        else:
            device = comfy.model_management.get_torch_device()

        # Initialize models on the correct device
        self.initialize_models(device)

        # Set random seed
        torch.manual_seed(seed)

        # Generate initial noise
        noise = torch.randn((batch_size, 4, height // 8, width // 8), 
                           dtype=torch.float32, device=device)

        # Transform the noise
        with torch.no_grad():
            if use_transformer:
                noise = self.noise_transformer(noise, residual=residual)
            if use_svd:
                noise = self.svd_noise_unet(noise, residual=residual)

        # Return in ComfyUI latent format
        return ({"samples": noise.cpu()},)
