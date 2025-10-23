import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

#########################
#    Embedding Layers   #
#########################

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]
  
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#########################
#      Causal Layer     #
#########################

class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)

    def forward_spda(
        self, x: torch.Tensor, mask: torch.Tensor = None, temp: float = 1.0
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor = None, temp: float = 1.0
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, temp: float = 1.0
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp)
        return self.forward_base(x, mask, temp)

class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))

class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor = None, attn_temp: float = 1.0
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp)
        x = x + self.mlp(x)
        return x

class CausalBlock(torch.nn.Module):
    causal_attn_mask: torch.Tensor

    def __init__(
        self,
        channels: int,
        data_dim: int,
        head_channels: int = 64,
        num_layers: int = 1,
        expansion: int = 4
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(1+data_dim, channels) * 1e-2)   # add 1 for cond_t token
        
        self.fwd_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_channels, expansion) for _ in range(num_layers)]
        )
        self.register_buffer('causal_attn_mask', torch.tril(torch.ones(1+data_dim, 1+data_dim)))

    def forward(
        self, x: torch.Tensor, cond_t: torch.Tensor = None
    ) -> torch.Tensor:
        x_in = x
        x = self.proj_in(x) + self.pos_embed
        x = x + cond_t if cond_t is not None else x

        for block in self.fwd_blocks:
            x = block(x, self.causal_attn_mask)

        return x + x_in


#########################
#     Readout Layer     #
#########################

class AggregationAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        head_channels: int,
        data_dim: int,
        expansion: int = 4
    ):
        super().__init__()

        self.data_dim = data_dim
        self.n_heads = channels // head_channels
        self.pos_embed = torch.nn.Parameter(torch.randn(data_dim, head_channels) * 1e-2)
        
        self.norm1 = torch.nn.LayerNorm(channels)
        self.attn_q = nn.Linear(channels, channels)
        self.attn_k = nn.Linear(channels, channels)
        self.attn_v = nn.Linear(channels, channels)

        self.attn_out = nn.Linear(channels, channels)
        self.norm2 = torch.nn.LayerNorm(channels)
        self.mlp = MLP(channels, expansion)

    def aggregate(
        self, combined_x: torch.Tensor, l2r_x: torch.Tensor, r2l_x: torch.Tensor
    ) -> torch.Tensor:
        all_embed = torch.cat([l2r_x, r2l_x], dim=1)  # B, 2D, E
        combined_x = self.norm1(combined_x)
        all_embed = self.norm1(all_embed)

        query = rearrange(
            self.attn_q(combined_x),
            "b s (h d) -> b s h d",
            h=self.n_heads,
        )
        key = rearrange(
            self.attn_k(all_embed),
            "b s (h d) -> b s h d",
            h=self.n_heads,
        )
        val = rearrange(
            self.attn_v(all_embed),
            "b s (h d) -> b s h d",
            h=self.n_heads,
        )

        key[:, :self.data_dim] = key[:, :self.data_dim] + rearrange(self.pos_embed, "s d -> 1 s 1 d")
        key[:, self.data_dim:] = key[:, self.data_dim:] + rearrange(self.pos_embed, "s d -> 1 s 1 d")
        
        query = query + rearrange(self.pos_embed, "s d -> 1 s 1 d")
        query = query / torch.sqrt(
            torch.tensor(query.shape[-1], device=query.device, dtype=torch.float32)
        )

        logits = torch.einsum("bqhd,bkhd->bhqk", query, key)  # B, H, D, 2*D
        att_l2r_mask = ~torch.triu(
            torch.ones((self.data_dim, self.data_dim), device=query.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)  # 1, D, D
        att_r2l_mask = ~torch.tril(
            torch.ones((self.data_dim, self.data_dim), device=query.device, dtype=torch.bool),
            diagonal=-1,
        ).unsqueeze(0)  # 1, D, D
        joint_mask = torch.cat([att_l2r_mask, att_r2l_mask], dim=-1).unsqueeze(
            0
        ) > 0  # 1, 1, D, 2*D
        
        attn_weights = torch.where(
            joint_mask,
            logits,
            torch.tensor(torch.finfo(logits.dtype).min, device=query.device),
        )  # B, H, D, 2*D
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        x = torch.einsum("bhqk,bkhd->bqhd", attn_weights, val)
        x = rearrange(x, "b s h d -> b s (h d)")
        x = self.attn_out(x)
        return x

    def forward(
        self, fwd_x: torch.Tensor, bwd_x: torch.Tensor, cond_t: torch.Tensor = None
    ) -> torch.Tensor:
        fwd_x = fwd_x[:, :-1, :]
        bwd_x = bwd_x[:, 1:, :]
        combined_x = (fwd_x + bwd_x) / math.sqrt(2)

        if cond_t is not None:
            combined_x = combined_x + cond_t
            fwd_x = fwd_x + cond_t
            bwd_x = bwd_x + cond_t

        x = combined_x + self.aggregate(combined_x, fwd_x, bwd_x)
        x = x + self.mlp(self.norm2(x))
        return x


#########################
#      Output Layer     #
#########################

class LEOutputLayer(nn.Module):
  def __init__(self, channels, vocab_size):
    super().__init__()

    self.vocab_size = vocab_size
    self.ommega_embedder = EmbeddingLayer(channels, vocab_size)
    self.norm_final = torch.nn.LayerNorm(channels)
 
  def forward(self, x, h, cond_t=None):
    h = self.norm_final(h)  # (b, d, h)
    if cond_t is not None:
       h = h + cond_t

    omega_emb = self.ommega_embedder(torch.arange(self.vocab_size, device=x.device))  # (vocab_size, h)
    omega_tau = omega_emb[None, None, ...]  # (1, 1, vocab_size, h)
    omega_xj = omega_emb[x].unsqueeze(-2)  # (b, d, 1, h)
    omega = omega_tau - omega_xj  # (b, d, vocab_size, h)
    
    Gt = torch.einsum("bdh,bdvh->bdv", h, omega)  # (b, d, vocab_size)
    return Gt


#####################################
#  Locally Equivariant Transformer  #
#####################################

class LEquiTFRtModel(nn.Module):
    def __init__(
        self,
        data_dim: int,
        vocab_size: int,
        hidden_dim: int,
        head_dim: int,
        num_blocks: int,
        layers_per_block: int,
    ):
        super().__init__()

        self.data_dim = data_dim
        self.vocab_size = vocab_size

        self.vocab_embedder = EmbeddingLayer(hidden_dim, vocab_size)
        self.time_embedder = TimestepEmbedder(hidden_dim)

        self.fwd_blocks = nn.ModuleList(
            [CausalBlock(hidden_dim, data_dim, head_dim, layers_per_block) for _ in range(num_blocks)]
        )
        self.bwd_blocks = nn.ModuleList(
            [CausalBlock(hidden_dim, data_dim, head_dim, layers_per_block) for _ in range(num_blocks)]
        )

        self.readout = AggregationAttentionBlock(hidden_dim, head_dim, data_dim)
        self.output_layer = LEOutputLayer(
            hidden_dim,
            vocab_size,
        )
        

    def forward(self, xt: torch.Tensor, cond_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the locally-equivariant transformer rate model.

        Args:
            xt (torch.Tensor): Input tensor of token IDs with shape (batch_size, data_dim).
            cond_t (torch.Tensor): Time condition tensor with shape (batch_size).

        Returns:
            torch.Tensor: Locally-equivariant rate matrix with shape (batch_size, data_dim, vocab_size).
        """

        x = self.vocab_embedder(xt)  # (batch_size, data_dim, hidden_dim)
        cond_t = self.time_embedder(cond_t).unsqueeze(1)    # (batch_size, 1, hidden_dim)

        fwd_x = torch.cat([cond_t, x], dim=1)
        for block in self.fwd_blocks:
            fwd_x = block(fwd_x) # (batch_size, 1+data_dim, hidden_dim)
        
        bwd_x = torch.cat([cond_t, x.flip(1)], dim=1)
        for block in self.bwd_blocks:
            bwd_x = block(bwd_x)  # (batch_size, 1+data_dim, hidden_dim)
        bwd_x = bwd_x.flip(1)

        output = self.readout(fwd_x, bwd_x, cond_t)
        output = self.output_layer(xt, output, cond_t)
        return output


if __name__ == "__main__":
    # python -m model

    seq_len = 16
    vocab_size = 27
    hidden_dim = 256
    head_dim = 128
    num_blocks = 3
    layers_per_block = 2
    bsz = 1
    device = torch.device("cuda")

    def debug():
        model = LEquiTFRtModel(seq_len, vocab_size, hidden_dim, head_dim, num_blocks, layers_per_block)
        model = model.to('cuda')
        model.eval()

        input = torch.randint(0, vocab_size, (bsz, seq_len)).to(device)
        cond_t = torch.rand((bsz,)).to(device)
        output = model(input, cond_t)

    def test_local_equivariance():
        # Fix random seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        model = LEquiTFRtModel(seq_len, vocab_size, hidden_dim, head_dim, num_blocks, layers_per_block)
        model = model.to('cuda')
        model.eval()

        base_tokens = torch.randint(0, vocab_size, (bsz, seq_len)).to(device)

        # Test local equivariance at each position
        for pos in range(seq_len):
            for tau in range(vocab_size):
                org_token = base_tokens[0, pos].item()
                if tau == org_token:
                    continue

                seq1 = base_tokens.clone()
                seq2 = base_tokens.clone()
                seq2[0, pos] = tau
                cond_t = torch.rand((bsz,)).to(device)
                # cond_t = None

                out1 = model(seq1, cond_t)
                out2 = model(seq2, cond_t)

                # Check that output at target_pos differs when changing token at pos
                diff = (out1[:, pos, tau] + out2[:, pos, org_token]).abs().item()
                print(f"Change at pos {pos}, org_token: {org_token}, tau: {tau}")
                print(f"Difference: {diff:.6f}, out1: {out1[:, pos, tau].item()}, out2: {out2[:, pos, org_token].item()}")
                assert diff < 1e-5, (
                    f"Output at {pos} with change to {tau} should satisfy local equivariance" + \
                    f" but got difference {diff:.6f}, out1: {out1[:, pos, org_token].item()}, out2: {out2[:, pos, tau].item()}"
                )
                print("Test passed!\n")

    def test_output_dependence():
        # Test that output at position i depends on tokens at other positions
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        model = LEquiTFRtModel(seq_len, vocab_size, hidden_dim, head_dim, num_blocks, layers_per_block)
        model = model.to('cuda')
        model.eval()

        base_tokens = torch.randint(0, vocab_size, (bsz, seq_len)).to(device)

        # Test dependence on each position
        for pos in range(seq_len):
            for target_pos in range(seq_len):
                if pos == target_pos:
                    continue

                seq1 = base_tokens.clone()
                seq2 = base_tokens.clone()
                seq2[0, pos] = (seq1[0, pos] + 1) % vocab_size
                cond_t = torch.rand((bsz,)).to(device)
                # cond_t = None

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out1 = model(seq1, cond_t)
                    out2 = model(seq2, cond_t)

                # Check that output at target_pos differs when changing token at pos
                diff = (out1[:, target_pos] - out2[:, target_pos]).abs().max().item()
                print(f"Change at pos {pos}, checking output at pos {target_pos}:")
                print(f"Difference: {diff:.6f}")
                assert diff > 1e-5, (
                f"Output at {target_pos} should depend on token at {pos}"
                )
                print("Test passed!\n")

    debug()

    print("Testing local equivariance...")
    test_local_equivariance()
    print("\nTesting output dependence...")
    test_output_dependence()
