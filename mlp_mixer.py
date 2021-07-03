"""Reference. https://github.com/rishikksh20/MLP-Mixer-pytorch"""
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(
        self, dim: int, num_patch: int, token_dim: int, channel_dim: int, dropout=0.0
    ):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n d -> b d n"),
            FeedForward(dim=num_patch, hidden_dim=token_dim, dropout=dropout),
            Rearrange("b d n -> b n d"),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, hidden_dim=channel_dim, dropout=dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)  # dim -> token_dim with skip connection
        x = x + self.channel_mix(x)  # token_dim => channel_dim with skip connection
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
        image_size: int,
        num_blocks: int,
        token_dim:int,
        channel_dim: int
        ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.patch_size = patch_size
        self.num_patch_oneway = (image_size // patch_size)
        self.num_patch = self.num_patch_oneway ** 2
        self.in_channels = in_channels
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(hidden_dim, self.num_patch, token_dim, channel_dim)
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, in_channels)

    def forward(self, input): # [B, in_channels, H, W]
        batch_size = input.size(0)
        output = self.patch_embedding(input) # [B, HXW, HIDDEN]
        for mixer in self.mixer_blocks: 
            output = mixer(output) # [B, HXW, HIDDEN]
        output = self.layer_norm(output)
        output = self.projection(output) # [B, HXW, IN_CHANNELS]
        output = output.view(batch_size, self.num_patch_oneway, self.num_patch_oneway, self.in_channels)
        output = self.spread(output)
        # output += input # broadcasting addition
        output *= input # broadcasting multiplication
        return output

    def spread(self, output: torch.Tensor):
        assert output.ndim == 4
        output = torch.repeat_interleave(output, self.patch_size, dim=1)
        output = torch.repeat_interleave(output, self.patch_size, dim=2)
        output = output.permute(0, 3, 1, 2) # [B, C, H, W]
        return output

        

# class MLPMixer(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_dim: int,
#         patch_size: int,
#         image_size: int,
#         num_blocks: int,
#         token_dim:int,
#         channel_dim: int
#         ):
#         super().__init__()
#         assert (
#             image_size % patch_size == 0
#         ), "Image dimensions must be divisible by the patch size."
#         self.patch_size = patch_size
#         self.num_patch_oneway = (image_size // patch_size)
#         self.num_patch = self.num_patch_oneway ** 2
#         self.in_channels = in_channels
#         self.patch_embedding = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=hidden_dim,
#                 kernel_size=patch_size,
#                 stride=patch_size,
#             ),
#             Rearrange("b c h w -> b (h w) c"),
#         )
#         self.mixer_blocks = nn.ModuleList(
#             [
#                 MixerBlock(hidden_dim, self.num_patch, token_dim, channel_dim)
#                 for _ in range(num_blocks)
#             ]
#         )
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.projection = nn.Linear(hidden_dim, in_channels)

#     def forward(self, input): # [B, in_channels, H, W]
#         batch_size = input.size(0)
#         output = self.patch_embedding(input) # [B, HXW, HIDDEN]
#         for mixer in self.mixer_blocks: 
#             output = mixer(output) # [B, HXW, HIDDEN]
#         output = self.layer_norm(output)
#         output = self.projection(output) # [B, HXW, IN_CHANNELS]
#         output = output.view(batch_size, self.num_patch_oneway, self.num_patch_oneway, self.in_channels)
#         output = self.spread(output)
#         output += input # broadcasting
#         return output

#     def spread(self, output: torch.Tensor):
#         assert output.ndim == 4
#         output = torch.repeat_interleave(output, self.patch_size, dim=1)
#         output = torch.repeat_interleave(output, self.patch_size, dim=2)
#         output = output.permute(0, 3, 1, 2) # [B, C, H, W]
#         return output
    
        

if __name__ == '__main__':
    in_channels = 3
    hidden_dim = 512
    patch_size = 8
    image_size = 512
    num_blocks = 1
    token_dim = 256
    channel_dim = 1024

    model = MLPMixer(
        in_channels,
        hidden_dim, 
        patch_size,
        image_size,
        num_blocks,
        token_dim,
        channel_dim,
        )
    sample = torch.rand(1, 3, 512, 512)
    model(sample)



