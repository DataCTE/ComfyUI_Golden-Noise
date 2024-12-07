import torch
from safetensors.torch import save_file
import os
import argparse

def convert_pth_to_safetensors(input_path, output_path=None):
    """Convert a .pth file to .safetensors format"""
    print(f"Loading {input_path}...")
    state_dict = torch.load(input_path, map_location='cpu')
    
    # If output path not specified, use same name but with .safetensors extension
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.safetensors'
    
    print(f"Converting and saving to {output_path}...")
    save_file(state_dict, output_path)
    print("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(description='Convert .pth files to .safetensors format')
    parser.add_argument('input', help='Input .pth file or directory')
    parser.add_argument('--output', help='Output .safetensors file or directory (optional)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Convert single file
        convert_pth_to_safetensors(args.input, args.output)
    elif os.path.isdir(args.input):
        # Convert all .pth files in directory
        for filename in os.listdir(args.input):
            if filename.endswith('.pth'):
                input_path = os.path.join(args.input, filename)
                if args.output:
                    output_path = os.path.join(args.output, filename.replace('.pth', '.safetensors'))
                else:
                    output_path = None
                convert_pth_to_safetensors(input_path, output_path)

if __name__ == "__main__":
    main()