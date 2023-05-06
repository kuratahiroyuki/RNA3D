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

#from rhofold.model.embedders_2 import MSAEmbedder, RecyclingEmbedder
#from rhofold.model.embedders_21 import MSAEmbedder, RecyclingEmbedder # for training_342, RNA-FM... encoding
from rhofold.model.embedders_22 import MSAEmbedder, RecyclingEmbedder # for training_342, nn.Embedding encodings instead of RNA-FM
from rhofold.model.e2eformer import E2EformerStack
from rhofold.model.structure_module_31 import StructureModule
from rhofold.model.heads_2 import DistHead, SSHead, pLDDTHead, TMScoreHead, compute_tm
from rhofold.utils.tensor_utils import add
from rhofold.utils import exists


class RhoFold(nn.Module):
    """The rhofold network"""

    def __init__(self, config):
        """Constructor function."""

        super().__init__()

        self.config = config

        self.msa_embedder = MSAEmbedder(
            **config.model.msa_embedder,
        )
        self.e2eformer = E2EformerStack(
            **config.model.e2eformer_stack,
        )
        self.structure_module = StructureModule(
            **config.model.structure_module,
        )
        self.recycle_embnet = RecyclingEmbedder(
            **config.model.recycling_embedder,
        )
        self.dist_head = DistHead(
            **config.model.heads.dist,
        )
        self.ss_head = SSHead(
            **config.model.heads.ss,
        )
        self.plddt_head = pLDDTHead(
            **config.model.heads.plddt,
        )
        ### from openfold
        self.tm_head = TMScoreHead(
            **config.model.heads.tm, #384, 64
        )

    def forward_cords(self, tokens, single_fea, pair_fea, frame_mask): 

        output = self.structure_module.forward(tokens, { "single": single_fea, "pair": pair_fea } )
        output['plddt'] = self.plddt_head(output['single'][-1]) #torch.Size([B, 1, 10, 384])
        #print('plddt {}'.format(output['plddt'] ))

        return output

    def forward_heads(self, pair_fea):

        output = {}
        output['ss'] = self.ss_head(pair_fea.float())
        output['p'], output['c4_'], output['n'] = self.dist_head(pair_fea.float())
        #print('ss {}'.format(output['ss'].shape)) # torch.Size([B, 1, 100, 100])
        #print('c4_ {}'.format(output['c4_'].shape)) # c4_ torch.Size([B, 40, 100, 100])

        ### from openfold
        output['tm'] = self.tm_head(pair_fea.float())

        output['ptm'] = compute_tm(output['tm'])

        return output

    def forward_one_cycle(self, tokens, rna_fm_tokens, frame_mask, all_atom_mask, recycling_inputs):
        '''
        Args:
            tokens: [bs, seq_len, c_z]
            rna_fm_tokens: [bs, seq_len, c_z]
        '''

        device = tokens.device

        msa_tokens_pert = tokens[:, :self.config.globals.msa_depth]
        #print('msa_tokens_pert:{}'.format(msa_tokens_pert))
        #print('msa_tokens_pert.shape:{}'.format(msa_tokens_pert.shape)) #torch.Size([B, 2, 23])

        msa_fea, pair_fea = self.msa_embedder.forward(tokens=msa_tokens_pert,
                                                      frame_mask = frame_mask,
                                                      rna_fm_tokens = rna_fm_tokens,
                                                      is_BKL=True)
        #print('msa_fea:{}'.format(msa_fea))
        #print('pair_fea:{}'.format(pair_fea))
        #print('msa_fea.shape:{}'.format(msa_fea.shape))
        #print('pair_fea.shape:{}'.format(pair_fea.shape))


        if exists(self.recycle_embnet) and exists(recycling_inputs):
            msa_fea_up, pair_fea_up = self.recycle_embnet(recycling_inputs['single_fea'],
                                                          recycling_inputs['pair_fea'],
                                                          recycling_inputs["cords_c1'"]) #sequence
            msa_fea[..., 0, :, :] += msa_fea_up
            pair_fea = add(pair_fea, pair_fea_up, inplace=False)

        msa_fea, pair_fea, single_fea = self.e2eformer(
            m=msa_fea,
            z=pair_fea,
            msa_mask=torch.ones(msa_fea.shape[:3]).to(device),
            pair_mask=torch.ones(pair_fea.shape[:3]).to(device),
            chunk_size=None,
        )
        #print('e2eformer msa_fea:{}'.format(msa_fea))
        #print('e2eformer pair_fea:{}'.format(pair_fea))#device='cuda:0'

        #print('e2eformer msa_fea.shape:{}'.format(msa_fea.shape)) # torch.Size([B, 1, 100, 256])
        #print('e2eformer pair_fea.shape:{}'.format(pair_fea.shape)) #torch.Size([B, 100, 100, 128])
        
        output = self.forward_cords(tokens, single_fea, pair_fea, frame_mask) # structure module
        
        #print('After structure output: {}'.format(output))

        output.update(self.forward_heads(pair_fea)) 
        #print('After structure output: {}'.format(output))
        
        
        recycling_outputs = {
            'single_fea': msa_fea[..., 0, :, :].detach(),
            'pair_fea': pair_fea.detach(),
            "cords_c1'": output["cords_c1'"][-1].detach(), #sequence
        }

        

        return output, recycling_outputs

    def forward(self,
                tokens,
                rna_fm_tokens,
                frame_mask,
                all_atom_mask,
                **kwargs):

        """Perform the forward pass.

        Args:

        Returns:
        """

        #tokens=tokens.squeeze(0) #strange
        #rna_fm_tokens=rna_fm_tokens.squeeze(0) #strange
        #print('tokens {}'.format(tokens.shape)) #torch.Size([B, 1, 1, 47]) <= [1,1,47]  
        #print('rna_fm_tokens {}'.format(rna_fm_tokens.shape)) #torch.Size([B, 1, 47]) <= [1,47]
        #print('seq {}'.format(seq))
        
        recycling_inputs = None
        output = None

        for _r in range(self.config.model.recycling_embedder.recycles):
            output, recycling_inputs = \
                self.forward_one_cycle(tokens, rna_fm_tokens, frame_mask, all_atom_mask, recycling_inputs)
            #print('recycling_inputs[cords_c1].shape: {}'.format( recycling_inputs["cords_c1'"].shape)) # torch.Size([B, 23, 3])
            #print(output)

        return output
