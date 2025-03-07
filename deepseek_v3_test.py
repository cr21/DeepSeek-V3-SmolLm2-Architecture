from deepseek_v3 import LlamaModel
import torch
import yaml

def test_deepseek_model():
    # Create a dummy configuration
    config = {
        'hidden_size': 576,
        'intermediate_size': 1536,
        'num_experts': 4,
        'num_shared_experts': 2,
        'top_k': 2,
        'vocab_size': 10000,
        'rope_theta': 10000.0,
        'init_method': {'std': 0.041666666666666664},
        'model_config': {
            'vocab_size': 10000,
            'hidden_size': 576,
            'rope_theta': 10000.0,
            'num_hidden_layers': 6,
            'tie_word_embeddings': True,
            'rms_norm_eps': 1e-5,
        }
    }
    with open("config_smollm2_135M.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Initialize the model
    model_config = config['model']
    print(model_config)
    model = LlamaModel(model_config)
    print(model)
    # Create dummy input
    batch_size = 2
    seq_length = 10
    print(model_config.keys())
    dummy_input = torch.randint(0, model_config['model_config']['vocab_size'], (batch_size, seq_length))

    # Forward pass
    logits, loss = model(dummy_input)

    # Check output shapes
    assert logits.shape == (batch_size * seq_length, model_config['model_config']['vocab_size']), "Logits shape mismatch"
    if loss is not None:
        assert loss.shape == (), "Loss shape mismatch"

    print("Test passed: Model is functioning correctly.")

# Uncomment to run the test
test_deepseek_model()
