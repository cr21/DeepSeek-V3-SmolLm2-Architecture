import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SiLU
import yaml


def _init_weights(module, std=0.041666666666666664):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)

class RotaryPositionalEmbedding(nn.Module):
    """
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L240
    Rotary Positional Embedding (RoPE) for transformers Implemntation derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H, D] or [B, T, D]
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: Output tensor with rotary positional embeddings applied.
        """
        # Handle different input shapes
        if len(x.shape) == 3:
            B, T, D = x.shape
            is_4d = False
        else:
            B, T, H, D = x.shape
            is_4d = True
            
        # For 3D tensors, we need to ensure D is even
        if not is_4d and D % 2 != 0:
            raise ValueError(f"Feature dimension {D} must be divisible by 2 for RoPE")

        # Generate position indices
        position = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(-1)

        # Generate frequencies
        if is_4d:
            # For 4D tensors, use the head dimension
            freqs = torch.exp(
                torch.arange(0, D, 2, dtype=torch.float32, device=x.device) * 
                -(torch.log(torch.tensor(self.theta)) / D)
            )
        else:
            # For 3D tensors, use the full dimension
            freqs = torch.exp(
                torch.arange(0, D, 2, dtype=torch.float32, device=x.device) * 
                -(torch.log(torch.tensor(self.theta)) / D)
            )

        # Compute sinusoids
        sinusoid = position * freqs
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        # Reshape sin and cos to match the input tensor's shape
        if is_4d:
            sin = sin.unsqueeze(0).unsqueeze(2)  # Shape: (1, T, 1, D // 2)
            cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, T, 1, D // 2)
        else:
            sin = sin.unsqueeze(0)  # Shape: (1, T, D // 2)
            cos = cos.unsqueeze(0)  # Shape: (1, T, D // 2)

        # Apply rotary embeddings
        x_rotated = x.clone()
        
        if is_4d:
            x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
            x_rotated[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin
        else:
            x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
            x_rotated[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin

        return x_rotated

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = self.config['num_attention_heads']
        self.hidden_size = self.config['hidden_size']
        # Ensure the hidden size is divisible by the number of attention heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.latent_dim = self.hidden_size // self.config['compression_ratio']

        # Matrix is decomposed into D and U matrix 
        # Compression KV Projection Matrix
        self.kv_proj_D = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        # Compression Q Projection Matrix
        self.q_proj_D = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        # UnCompression k projection matrix
        self.k_proj_U = nn.Linear(self.latent_dim, self.hidden_size//2, bias=False)
        # UnCompression v projection matrix
        self.v_proj_U = nn.Linear(self.latent_dim, self.hidden_size, bias=False)    
        # UnCompression Q projection matrix
        self.q_proj_U = nn.Linear(self.latent_dim, self.hidden_size//2, bias=False)

        # Rope Key Components, K is built from X and Q is build from q_proj_D
        self.rope_k = nn.Linear(self.hidden_size, self.hidden_size//2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_size//2, bias=False)
        # output projection matrix
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(self.hidden_size//2, self.config['rope_theta'])

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # Batch Size, Sequence Length, Hidden Size
        # Compression KV Projection Matrix
        kv_d = self.kv_proj_D(x) # [B, T, Latent Dim]
        # Compression Q Projection Matrix
        q_d = self.q_proj_D(x) # [B, T, Latent Dim]
        # Uncompress KV & Q Projection Matrix
        k_proj_2 = self.k_proj_U(kv_d) # [B, T, Hidden Size//2]
        q_proj_2 = self.q_proj_U(q_d) # [B, T, Hidden Size//2]
        v = self.v_proj_U(kv_d) # [B, T, Hidden Size]
        
        # Rope components
        k_rope_2 = self.rope_k(x) # [B, T, Hidden Size//2]
        q_rope_2 = self.rope_q(q_d) # [B, T, Hidden Size//2]
        
        # Apply ROPE to the rope components
        k_rope_2 = self.rotary_emb(k_rope_2, T) # [B, T, Hidden Size//2]
        q_rope_2 = self.rotary_emb(q_rope_2, T) # [B, T, Hidden Size//2]
        
        # Reshape Components for Multi-Head Attention
        k_proj_2 = k_proj_2.view(B, T, self.num_attention_heads, self.head_dim//2)
        k_rope_2 = k_rope_2.view(B, T, self.num_attention_heads, self.head_dim//2)
        q_proj_2 = q_proj_2.view(B, T, self.num_attention_heads, self.head_dim//2)
        q_rope_2 = q_rope_2.view(B, T, self.num_attention_heads, self.head_dim//2)
        
        # Concatenate Components
        k = torch.cat((k_proj_2, k_rope_2), dim=-1) # [B, T, H, D]
        q = torch.cat((q_proj_2, q_rope_2), dim=-1) # [B, T, H, D]
        v = v.view(B, T, self.num_attention_heads, self.head_dim)
        
        # Reshape Components for Multi-Head Attention
        k = k.transpose(1, 2) # [B, H, T, D]
        q = q.transpose(1, 2) # [B, H, T, D]
        v = v.transpose(1, 2) # [B, H, T, D]
        
        # Apply Scaled Dot-Product Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, 
                                                  dropout_p=0.0, 
                                                  is_causal=True,
                                                  attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        return self.o_proj(attn_out) # [B, T, C]

class DeepSeekExpertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMOE(nn.Module):
    """
    A Mixture of Experts (MoE) layer that routes input through a set of expert layers.

    This class implements a mixture of experts mechanism where a subset of experts is selected 
    for each input token based on learned routing logits. The output is a combination of the 
    shared experts and the routed experts, allowing for efficient computation and increased 
    model capacity.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        intermediate_size (int): The size of the intermediate layer.
        num_experts (int): Total number of experts available.
        num_shared_experts (int): Number of shared experts that are used for all inputs.
        top_k (int): The number of top experts to route each input to.
        shared_experts (nn.ModuleList): List of shared expert layers.
        routed_experts (nn.ModuleList): List of routed expert layers.
        routing_fn (nn.Linear): Linear layer for computing routing logits.
        routing_bias (nn.Parameter): Bias for the routing logits.

    Methods:
        forward(x): Forward pass through the MoE layer, routing input through selected experts.
    """
    def __init__(self, hidden_size, intermediate_size, num_experts, num_shared_experts, top_k):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.num_routed_experts = num_experts - num_shared_experts
        self.shared_experts = nn.ModuleList(
            [DeepSeekExpertLayer(self.hidden_size, self.intermediate_size) for _ in range(self.num_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [DeepSeekExpertLayer(self.hidden_size, self.intermediate_size) for _ in range(self.num_routed_experts)]
        )
        
        # Routing Function
        self.routing_fn = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
    def forward(self, x):
        B, T, C = x.size()
        shared_out = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts>1:
            shared_out = shared_out/self.num_shared_experts # normalize the shared experts
        # calculate the routing function
        routing_logits = self.routing_fn(x) + self.routing_bias # [B, T, num_routed_experts]
        # GEt Topk Experts per token
        routing_probs = torch.sigmoid(routing_logits) # [B, T, num_routed_experts]
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1) # [B, T, top_k]
        # normalize the top k scores
        scores  = scores/torch.sum(scores, dim=-1, keepdim=True)
        # process the routed experts
        #combined_output = torch.zeros(B, T, C, device=x.device)
        combined_output = torch.zeros_like(x)

        # Calculate expert load for all experts
        expert_load = torch.zeros(self.num_routed_experts, device=x.device)
        for i in range(self.top_k):
            expert_idx = indices[:, :, i] # [B, T, top_k]
        
            expert_scores = scores[...,i:i+1]
            # process the routed experts
            for j in range(self.num_routed_experts):
                mask = (expert_idx == j) # [B, T, 1]
                if mask.any():
                    # Track expert usage (load)
                    expert_load[j] += mask.sum().float() / (B * T * self.top_k)
                    # Process tokens through this expert
                    expert_input = x[mask] # [B, T, 1, C]
                    expert_output = self.routed_experts[j](expert_input)
                    combined_output[mask] += expert_scores[mask] * expert_output
        final_output = shared_out + combined_output
        router_z_loss = self.update_bias_terms(expert_load)
        return final_output, router_z_loss
    
    def update_bias_terms(self, expert_load, router_z_loss_coef=0.001):
        # Balance expert routing by adjusting the bias terms
        # Target load is uniform distribution across experts
        target_load = 1.0 / self.num_routed_experts
        
        # Calculate load imbalance for each expert
        load_diff = expert_load - target_load
        
        # Dynamic update rate based on the magnitude of imbalance
        # Larger imbalances get larger corrections
        update_rate = 0.1 * torch.abs(load_diff)
        
        # Update the routing bias to counteract imbalance
        # Decrease bias for overutilized experts, increase for underutilized
        self.routing_bias.data -= update_rate * load_diff
        
        # Calculate the router z-loss to discourage extreme routing probabilities
        # This helps stabilize training without auxiliary losses
        # Z-loss encourages routing probabilities to stay away from 0 and 1
        router_z_loss = router_z_loss_coef * torch.mean(torch.log(torch.sum(
            torch.exp(self.routing_fn.weight), dim=-1)))
        
        return router_z_loss

    def update_bias_terms_old(self, expert_load, ):
        # adjust the bias terms based on the expert load
        target_load = 1/self.num_experts
        load_diff = expert_load - target_load
        # dyanamic update the bias based on the load imbalance
        update_rate = 0.1 * torch.abs(load_diff)
        # dyanmic update the bias terms using update rate
        self.routing_bias = self.routing_bias - update_rate * load_diff

        # for i in range(self.num_routed_experts):
        #     if expert_load[i] < target_load:
        #         self.routing_bias[i] -= 1
        #     else:
        #         self.routing_bias[i] += 1
class LlamaMLP(nn.Module):
    """
    (mlp): LlamaMLP(
        (moe): DeepSeekMOE(
          (shared_experts): ModuleList(
            (0): DeepSeekExpertLayer(
              (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
              (up_proj): Linear(in_features=576, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=576, bias=False)
              (act_fn): SiLU()
            )
          )
          (routed_experts): ModuleList(
            (0-2): 3 x DeepSeekExpertLayer(
              (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
              (up_proj): Linear(in_features=576, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=576, bias=False)
              (act_fn): SiLU()
            )
          )
          (routing_fn): Linear(in_features=576, out_features=3, bias=False)
        )
      )
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.moe = DeepSeekMOE(hidden_size=config['hidden_size'],
                                intermediate_size=config['intermediate_size'],
                                num_experts=config['num_experts'],
                                num_shared_experts= config['num_shared_experts'], 
                                top_k=config['top_k'])
        # self.gate_proj = nn.Linear(self.config['hidden_size'], self.config['intermediate_size'], bias=False)
        # self.up_proj = nn.Linear(self.config['hidden_size'], self.config['intermediate_size'], bias=False)
        # self.down_proj = nn.Linear(self.config['intermediate_size'], self.config['hidden_size'], bias=False)
        # self.act_fn = SiLU()
    def forward(self, x):
        output, router_z_loss = self.moe(x)
        return output, router_z_loss
        # gate = self.gate_proj(x)
        # up = self.up_proj(x)
        # down = self.down_proj(self.act_fn(gate)*up)
        # return down 
    
class LlamaRMSNorm(nn.Module):
    """
    (norm): LlamaRMSNorm((576,), eps=1e-05)
        # RMSNorm Formula:
        #    RMS(x) = sqrt((1 / d) * sum(x_i^2 for i in range(d)))
        #    x_normalized = x / RMS(x)
        #    output = gamma * x_normalized
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = self.config['rms_norm_eps']
        self.weight = nn.Parameter(torch.ones(self.config['hidden_size']))
    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return  self.weight *rms * x
    
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadLatentAttention(self.config)
        self.input_layernorm = LlamaRMSNorm(self.config)
        self.mlp = LlamaMLP(self.config)
        self.post_attention_layernorm = LlamaRMSNorm(self.config)   
    
    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x, router_z_loss = self.mlp(x)
        x = x + residual
        return x, router_z_loss
    
class DeepSeekV3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.init_method = config['init_method']
        self.config = config['model_config']
        self.embed_tokens = nn.Embedding(self.config['vocab_size'], self.config['hidden_size'])
        self.rotary_emb = RotaryPositionalEmbedding(self.config['hidden_size'], self.config['rope_theta'])
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config) for _ in range(self.config['num_hidden_layers'])])
        self.norm = LlamaRMSNorm(self.config)
        self.lm_head = nn.Linear(self.config['hidden_size'], self.config['vocab_size'], bias=False)
        
        if self.config['tie_word_embeddings']:
            self.lm_head.weight = self.embed_tokens.weight
        
        self.apply(lambda m: _init_weights(m, self.init_method['std']))
    
    def forward(self, x, y=None):
        x = self.embed_tokens(x)
        total_router_z_loss = 0.0
        for layer in self.layers:
            x, router_z_loss = layer(x)
            total_router_z_loss += router_z_loss
        x = self.norm(x)
        logits = self.lm_head(x) # B,T,V
        logits = logits.view(-1, logits.size(-1))  # Shape: [B*T, V] # 20, 49152
        if y is not None:
            y = y.view(-1)  # Shape: [B*T] # 20
            ce_loss = torch.nn.functional.cross_entropy(logits, y)
            # Combine CE loss with router z-loss
            loss = ce_loss + total_router_z_loss
            return logits, loss
        else:
            return logits, None

    
    def generate(self, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None):
        model = self.to(device)
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

# if __name__ == "__main__":
#     torch.manual_seed(0)
#     config = yaml.load(open("config_smollm2_135M.yaml", "r"), Loader=yaml.FullLoader)
#     print(config.keys())
#     model_config = config['model']['model_config']
#     print(model_config)
#     model = DeepSeekV3Model(config['model'])
#     x_tokens = torch.randint(0, model_config['vocab_size'], (1, 10))  # Generate random token indices
#     print(model(x_tokens).shape)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params}") #134515008
#     print(model)