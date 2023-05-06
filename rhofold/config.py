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

import ml_collections as mlc

rhofold_config = mlc.ConfigDict(
    {
        "max_len_seq": 20, ###
        "globals": {
            "c_z": 128,
            "c_m": 256,
            "c_t": 64,
            "c_e": 64,
            "c_s": 384,
            'msa_depth': 128,
            'frame_version': 'v5.0',
            "eps": 1e-8,
        },
        ###
        "data_module": {
            "use_small_bfd": False,
            "data_loaders": {
                "batch_size": 8, #8
                "num_workers": 0, #0 16
                "pin_memory": True, #True
                "shuffle": False, # False
            },
        },
        "model": {
            "input_embedder": {
                "tf_dim": 22,
                "msa_dim": 49,
                "c_z": 128,
                "c_m": 256,
                "relpos_k": 32,
            },
            'msa_embedder':{
                "c_z": 128,
                "c_m": 256,
                'rna_fm':{
                    'enable': True,
                },
            },
            "recycling_embedder": {
                'recycles': 3,
                "c_z": 128,
                "c_m": 256,
                "min_bin": 2,
                "max_bin": 40,
                "no_bins": 40,
            },
            "e2eformer_stack": {
                "blocks_per_ckpt": 1,
                "c_m": 256,
                "c_z": 128,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": 384,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 12,
                "transition_n": 4,
            },
            "structure_module": {
                "c_s": 384,
                "c_z": 128,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 6,
                "trans_scale_factor": 10,
                'refinenet':{
                    'enable': True,
                    'dim': 64,
                    'is_pos_emb': True,
                    'n_layer': 4,
                }
            },
            "heads": {
                "plddt": {
                    "c_in": 384,
                    "no_bins": 50,
                },
                "dist": {
                    "c_in": 128,
                    "no_bins": 40, #40
                },
                "ss": {
                    "c_in": 128,
                    "no_bins": 1,
                },
                "tm": { ###
                    "c_z": 128,
                    "no_bins": 64,
                },
            },
        },
        ###
        "backbone": {
            "weight" : 1
        },

        "fape": {
            "backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5,
            },
            "sidechain": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "weight": 0.5,
            },
            "eps": 1e-4,
            "weight": 1.0,
        },
        # addition
        "rmsd": { ###
            "backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5,
            },
            "sidechain": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "weight": 0.5,
            },
            "eps": 1e-4,
            "weight": 1.0,
        },
        "plddt_loss": {
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "cutoff": 15.0,
            "no_bins": 50,
            "eps": 1e-10,
            "weight": 0.01,
        },
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 40, #64
            "eps": 1e-6,
            "weight": 0.3,
        },
        "tm": { ###
            "max_bin": 31,
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "no_bins": 64,
            "eps": 1e-6,
            "weight": 1,
        },
    }
)



