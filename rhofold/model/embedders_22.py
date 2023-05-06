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
from typing import Tuple

from rhofold.model.primitives import Linear, LayerNorm
from rhofold.utils.tensor_utils import add
import rhofold.model.rna_fm as rna_esm
import rhofold.model.fm as fm
from rhofold.model.msa import MSANet
from rhofold.model.pair import PairNet
from rhofold.utils import exists
from rhofold.utils.alphabet import RNAAlphabet

class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = 1e8

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update

class MSAEmbedder(nn.Module):
    """MSAEmbedder """

    def __init__(self,
                 c_m,
                 c_z,
                 rna_fm=None,
                 ):
        super().__init__()

        #print('rna_fm {}'.format(rna_fm)) #enable: true

        self.rna_fm, self.rna_fm_reduction = None, None
        self.embed_tokens, self.embed_tokens_reduction = None, None 

        self.mask_rna_fm_tokens = False

        self.alphabet = RNAAlphabet.from_architecture('RNA')

        self.msa_emb = MSANet(d_model = c_m,
                               d_msa = len(self.alphabet),
                               padding_idx = self.alphabet.padding_idx,
                               is_pos_emb = True,
                               )

        self.pair_emb = PairNet(d_model = c_z,
                                 d_msa = len(self.alphabet),
                                 )

        if exists(rna_fm) and rna_fm['enable']:
            # Load RNA-FM model
            self.rna_fm_dim = 640
            #self.rna_fm, _ = rna_esm.pretrained.esm1b_rna_t12() ###
            self.rna_fm, _ = fm.pretrained.rna_fm_t12() ###
            self.rna_fm.eval()
            #self.rna_fm.train()
            for param in self.rna_fm.parameters():
                param.detach_()
            self.rna_fm_reduction = nn.Linear(self.rna_fm_dim + c_m, c_m)
        else :
            #print(f'self.alphabet = {self.alphabet}' )
            #print(len(self.alphabet)) #17
            self.embed_tokens_dim = 640   ###        
            self.embed_tokens = nn.Embedding(20, self.embed_tokens_dim) ### vocab size = 20
            self.embed_tokens.eval()
            for param in self.embed_tokens.parameters():
                param.detach_()
            self.embed_tokens_reduction = nn.Linear(self.embed_tokens_dim + c_m, c_m)

    def forward(self, tokens, frame_mask, rna_fm_tokens = None, is_BKL = True, **unused):

        #print('tokens {}'.format(tokens))
        #print('tokens.ndim {}'.format(tokens.ndim))

        assert tokens.ndim == 3
        if not is_BKL:
            tokens = tokens.permute(0, 2, 1)

        B, K, L = tokens.size() # batch_size, num_alignments, seq_len
        #print('tokens: {} '.format(tokens))
        #print('BKL: {} {} {}'.format(B, K, L)) #1 ,1, 23
        #print('self.alphabet: {} '.format(self.alphabet))
        msa_fea = self.msa_emb(tokens)

        #print("exists(self.rna_fm) {}".format(self.rna_fm) )

       
        if exists(self.rna_fm) : 
            results = self.rna_fm(rna_fm_tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)
            #results = results["representations"][12]            
            #print('results {}'.format(results)) #no grad_fn
            #token_representations = results.unsqueeze(1).expand(-1, K, -1, -1)  
            #print(results)
            token_representations = results["representations"][12].unsqueeze(1).expand(-1, K, -1, -1)  
            #token_representations = (results["representations"][12]*frame_mask[...,None]).unsqueeze(1).expand(-1, K, -1, -1) 
            #print('msa_fea: {}'.format(msa_fea.shape))#torch.Size([8, 1, 20, 256]) 
            #print('token_representations: {}'.format(token_representations.shape)) #torch.Size([8, 1, 20, 640])
            msa_fea = self.rna_fm_reduction(torch.cat([token_representations, msa_fea], dim = -1))
        else : 
            results = self.embed_tokens(rna_fm_tokens) 
            token_representations = results.unsqueeze(1).expand(-1, K, -1, -1) ###
            #print('msa_fea: {}'.format(msa_fea.shape))#torch.Size([8, 1, 20, 256]) 
            #print('token_representations: {}'.format(token_representations.shape))# torch.Size([8, 1, 20, 640])    
            #print(torch.cat([token_representations, msa_fea], dim=-1))
            msa_fea = self.embed_tokens_reduction(torch.cat([token_representations, msa_fea], dim=-1)) ###
            
    
        pair_fea = self.pair_emb(tokens, t1ds = None, t2ds = None)
        #print('pair_fea: {}'.format(pair_fea.shape)) #[8, 20, 20, 128]
        #square_mask = frame_mask[:,:,None]*frame_mask[:,None,:]
        #pair_fea = pair_fea * square_mask[:,:,:,None]
        
        #print('pair_fea: {}'.format(pair_fea)) 
        #print('msa_fea: {}'.format(msa_fea.shape))#torch.Size([8, 1, 20, 256])

        return msa_fea, pair_fea
