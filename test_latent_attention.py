import torch
import yaml
from deepseek_v3 import MultiHeadLatentAttention

def test_latent_attention():
    # Load configuration
    with open("config_smollm2_135M.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Extract model configuration
    model_config = config['model']['model_config']
    
    # Create an instance of MultiHeadLatentAttention
    latent_attn = MultiHeadLatentAttention(model_config)
    
    # Print model information
    print(f"Model configuration:")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Attention heads: {model_config['num_attention_heads']}")
    print(f"  Compression ratio: {model_config['compression_ratio']}")
    print(f"  Latent dimension: {model_config['hidden_size'] // model_config['compression_ratio']}")
    
    # Generate random input
    batch_size = 2
    seq_len = 16
    hidden_size = model_config['hidden_size']
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    print(f"\nInput shape: {x.shape}")
    
    # Run the model
    output = latent_attn(x)
    
    print(f"Output shape: {output.shape}")
    
    # Check if output shape matches input shape
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    print("\nTest passed! Output shape matches input shape.")
    
    # Print some statistics about the output
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")

if __name__ == "__main__":
    test_latent_attention() 