# Install PyTorch with CUDA so training/inference can use GPU.
# Run from backend root: .\scripts\install_pytorch_gpu.ps1
# Python 3.14: use cu126 (no cu121 wheels). Set: $env:CUDA_VERSION = "cu126"
# Optional: $env:CUDA_VERSION = "cu118" | "cu121" | "cu124" | "cu126" | "cu128"

$cu = if ($env:CUDA_VERSION) { $env:CUDA_VERSION } else { "cu126" }
$index = "https://download.pytorch.org/whl/$cu"
Write-Host "Installing PyTorch + torchvision with $cu (index: $index)"
pip uninstall -y torch torchvision 2>$null
pip install torch torchvision --index-url $index
Write-Host 'Done. Verify with: python -c "import torch; print(torch.cuda.is_available())"'