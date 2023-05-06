# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from rhofold.utils import default
from rhofold.model.primitives_of import Linear
from typing import Dict, Optional, Tuple

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2

        return x

class FeedForwardLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 p_drop = 0.1,
                 d_model_out = None,
                 is_post_act_ln = False,
                 **unused,
                 ):

        super(FeedForwardLayer, self).__init__()
        d_model_out = default(d_model_out, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.post_act_ln = LayerNorm(d_ff) if is_post_act_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model_out)
        self.activation = nn.ReLU()

    def forward(self, src):
        src = self.linear2(self.dropout(self.post_act_ln(self.activation(self.linear1(src)))))
        return src


class DistHead(nn.Module):
    def __init__(self,
                 c_in,
                 no_bins=64, #40
                 **kwargs):
        super(DistHead, self).__init__()
        self.norm = LayerNorm(c_in)
        self.proj = nn.Linear(c_in, c_in)

        self.resnet_dist_0 = FeedForwardLayer(d_model=c_in, d_ff=c_in * 4, d_model_out=no_bins,
                                            **kwargs)
        self.resnet_dist_1 = FeedForwardLayer(d_model=c_in, d_ff=c_in * 4, d_model_out=no_bins,
                                            **kwargs)
        self.resnet_dist_2 = FeedForwardLayer(d_model=c_in, d_ff=c_in * 4, d_model_out=no_bins,
                                            **kwargs)

    def forward(self, x):

        x = self.norm(x)
        x = self.proj(x)

        logits_dist0 = self.resnet_dist_0(x).permute(0, 3, 1, 2) ###permuteの理由は
        logits_dist1 = self.resnet_dist_1(x).permute(0, 3, 1, 2)
        logits_dist2 = self.resnet_dist_2(x).permute(0, 3, 1, 2)

        #print('logits_dist0 {}'.format(logits_dist0[0,1,1,:])) #logits_dist0 torch.Size([B, 40, 20, 20]) B bin L L

        #print('logits_dist1 {}'.format(logits_dist1[0,1,1,:]))
        #print('logits_dist2 {}'.format(logits_dist2[0,1,1,:]))

        return logits_dist0, logits_dist1, logits_dist2


class SSHead(nn.Module):
    def __init__(self,
                 c_in,
                 no_bins=1,
                 **kwargs):
        super(SSHead, self).__init__()
        self.norm = LayerNorm(c_in)
        self.proj = nn.Linear(c_in, c_in)
        self.ffn = FeedForwardLayer(d_model=c_in, d_ff = c_in*4, d_model_out=no_bins, **kwargs)

    def forward(self, x):

        x = self.norm(x)
        x = self.proj(x)
        x = 0.5 * (x + x.permute(0, 2, 1, 3))
        logits = self.ffn(x).permute(0, 3, 1, 2)

        return logits

class pLDDTHead(nn.Module):
    def __init__(self, c_in, no_bins = 50):
        super(pLDDTHead, self).__init__()

        self.bin_vals = (torch.arange(no_bins).view(1, 1, -1) + 0.5) / no_bins

        self.net_lddt = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, c_in),
            nn.ReLU(),
            nn.Linear(c_in, no_bins),
        )
        self.sfmx = nn.Softmax(dim=2)

    def forward(self, sfea_tns):

        logits = self.net_lddt(sfea_tns)

        self.bin_vals = self.bin_vals.to(logits.device)
        #print('self.bin_vals {}'.format(self.bin_vals)) #  torch.Size([B, 1, 50]) #no_bins
        #print('self.bin_vals {}'.format(self.bin_vals.shape)) 

        plddt_local = torch.sum(self.bin_vals * self.sfmx(logits), dim=2) #[B,L]
        #print('plddt_local {}'.format(plddt_local)) #tensor([[0.7350, 0.7920, 0.7617, 0.7474, 0.5781, 0.6438, 0.5897, 0.6902, 0.7607, 0.7701]]
        #print('self.bin_vals * self.sfmx(logits) {}'.format((self.bin_vals * self.sfmx(logits)).shape)) #  torch.Size([B, L, 50]) #no_bins

        plddt_global = torch.mean(plddt_local, dim=1)

        #return  plddt_local, plddt_global, self.bin_vals * self.sfmx(logits) ###
        return plddt_local, plddt_global, logits

### from openfold ###
def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    torch.sum(residue_weights)
    n = logits.shape[-2]
    clipped_n = max(n, 19)

    #d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8
    d0 = 0.6 * (clipped_n - 0.5) ** (1.0 / 2) - 2.5

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights
    
    #print('weighted in compute_tm {}'.format(weighted))
    argmax = (weighted == torch.max(weighted)).nonzero()[0] #IndexError: index 0 is out of bounds for dimension 0 with size 0

    return per_alignment[tuple(argmax)]


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """
    #def __init__(self, c_z, no_bins, **kwargs):
    def __init__(self, c_z, no_bins):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        #print('tm z {}'.format(z.shape)) #Torch.Size([B, 100, 100, 128])
        logits = self.linear(z)
        #print('tm logits {}'.format(logits.shape))
        
        return logits

