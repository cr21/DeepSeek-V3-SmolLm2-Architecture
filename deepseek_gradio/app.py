import gradio as gr
import torch
from transformers import AutoTokenizer
import yaml
from deepseek_v3 import DeepSeekV3Model
import os


def generate_helper(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None):
    
    model = model.to(device)
    idx = idx.to(device)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits, _ = model(idx_cond)  # Unpack both logits and loss (ignore loss)
            logits = logits.view(idx_cond.shape[0], -1, model.config['vocab_size'])  # Reshape to [batch, seq, vocab]
            
        # Get the logits for the last token only
        logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        
        if top_k is not None:
            # top k sampling
            top_logits, top_pos = torch.topk(logits, top_k)
            min_logit = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_logit,
                               torch.tensor(float('-inf')).to(logits.device),
                               logits)
        
        # temperature scaling
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        if idx_next.item() == eos_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx

def get_config(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config

def extract_and_save_weights(config_path, checkpoint_path, weights_path, device):
    """Extract model weights from checkpoint and save as a separate .pt file"""
    print(f"Extracting weights from checkpoint: {checkpoint_path}")
    config = get_config(config_path)
    model = DeepSeekV3Model(config['model'])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Save just the model weights
    torch.save(state_dict, weights_path)
    print(f"Model weights saved to: {weights_path}")
    return state_dict

def load_weights(config, weights_path, device):
    """Load model from weights file"""
    print(f"Loading model from weights: {weights_path}")
    model = DeepSeekV3Model(config['model'])
    state_dict = torch.load(weights_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    return model

def get_tokenizer(config):
    tokenizer_path = config['tokenizer']['tokenizer_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    return tokenizer, vocab_size

def generate_text(model, tokenizer, input_text, max_new_tokens, context_length, temperature, top_k, eos_token, device):
    encoded_text = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_text = generate_helper(model, 
                            idx=encoded_text,
                            max_new_tokens=max_new_tokens,
                            context_length=context_length, 
                            temperature=temperature, 
                            top_k=top_k, 
                            eos_token=eos_token, 
                            device=device)
    return tokenizer.decode(generated_text.squeeze(0))



# Initialize model and tokenizer
def initialize_model():
    config_path = "config_smollm2_135M.yaml"
    # Use HF Hub or another external storage instead of local path
    model_id = "crpatel/DeepSeek-V3-SmolLm2"  # Replace with your actual model ID
    weights_path = "model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration
    config = get_config(config_path)
    
    # Check if weights exist locally, otherwise download from HF Hub
    if not os.path.exists(weights_path):
        try:
            from huggingface_hub import hf_hub_download
            print(f"Downloading model weights from Hugging Face Hub: {model_id}")
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename="model.pt"
            )
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Falling back to local checkpoint extraction if available")
            checkpoint_path = "checkpoints/model_100000_step_avg_loss_4.61663.pth"
            if os.path.exists(checkpoint_path):
                extract_and_save_weights(config_path, checkpoint_path, weights_path, device)
            else:
                raise FileNotFoundError(f"Neither weights file nor checkpoint found. Please upload model to HF Hub first.")
    
    # Load model from weights
    model = load_weights(config, weights_path, device)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer, vocab_size = get_tokenizer(config)
    
    return model, tokenizer, device

def generate_response(prompt, max_new_tokens):
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text=prompt,
        max_new_tokens=max_new_tokens,
        context_length=256,
        temperature=0.9,
        top_k=2,
        eos_token=tokenizer.eos_token_id,
        device=device
    )
    return generated_text

# Initialize global variables
model, tokenizer, device = initialize_model()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(
            lines=3, 
            placeholder="Enter your prompt here...",
            label="Input Prompt"
        ),
        gr.Slider(
            minimum=50,
            maximum=256,
            value=100,
            step=10,
            label="Max New Tokens"
        )
    ],
    outputs=gr.Textbox(
        lines=5,
        label="Generated Text"
    ),
    title="DeepSeek-V3 Text Generator",
    description="Enter a prompt and adjust the maximum number of tokens to generate text with DeepSeek-V3 SmolLM2  model."
)

if __name__ == "__main__":
    iface.launch()