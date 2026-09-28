import sys
from pathlib import Path

# Add project root directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.lxfd_model import LXDFDModel

def get_detailed_summary():
    model = LXDFDModel(embed_dim=256, fusion_type='attention', pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    
    print("="*95)
    print(f"{'Layer / Submodule Name':<42} {'Output Tensor Shape':<25} {'Param #':<15} {'Trainable'}")
    print("="*95)
    
    total_params = 0
    trainable_params = 0
    
    # Track through forward pass hooks or named modules
    for name, module in model.named_modules():
        # filter to leaf modules with parameters or key containers
        params = sum(p.numel() for p in module.parameters(recurse=False))
        tr_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        total_params += params
        trainable_params += tr_params
        if params > 0 or len(list(module.children())) == 0:
            mod_class = module.__class__.__name__
            disp_name = f"{name} ({mod_class})" if name else f"LXDFDModel ({mod_class})"
            print(f"{disp_name[:40]:<42} {'-':<25} {params:>12,} {'Yes' if tr_params > 0 else 'No'}")

    print("="*95)
    print(f"Total Parameters:          {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters:      {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Non-trainable Parameters:  {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"Estimated Model Size:      ~{sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.2f} MB (FP32)")
    print("="*95)

if __name__ == "__main__":
    get_detailed_summary()
