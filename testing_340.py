""" """
import logging
import os
import sys
import pickle
import random
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import optim

from rhofold.data.balstn import BLASTN
from rhofold.rhofold_325 import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet_33 import get_features

from rhofold.utils.loss_rho_281 import RhoFoldLoss
from rhofold.utils.rigid_utils import Rotation, Rigid
from rhofold.utils.constants import RNA_CONSTANTS


def pdb_data_input( coords, config, max_len):

    batch=[]        
    for i, (idx, seq_coord) in enumerate(coords.items()):
        data = {}
        pdb_id = idx
        seq = seq_coord['seq']
        coord = seq_coord['3d_coord']

        #print(f'pdb_id: {pdb_id}')
        n_seq = len(seq)
        #print(n_seq)
        seq = seq + '-'*(max_len-n_seq)
        #print(seq)

        all_atom_positions = torch.tensor(coord)
        all_atom_positions = torch.cat((all_atom_positions, torch.zeros([max_len-n_seq, RNA_CONSTANTS.ATOM_NUM_MAX, 3])), dim=0)
        #print(all_atom_positions )
        #print(all_atom_positions.shape)

        all_atom_mask = torch.ones([n_seq, RNA_CONSTANTS.ATOM_NUM_MAX])
        for res_id in range(n_seq):
            if seq[res_id] == 'A' :
                all_atom_mask[res_id, 22] = 0
            elif seq[res_id]  == 'G':
                pass
            elif seq[res_id]  == 'U':
                all_atom_mask[res_id, 20:RNA_CONSTANTS.ATOM_NUM_MAX] = 0
            elif seq[res_id] == 'C':
                all_atom_mask[res_id, 20:RNA_CONSTANTS.ATOM_NUM_MAX] = 0
            else:
                pass
        all_atom_mask = torch.cat((all_atom_mask, torch.zeros([max_len-n_seq, RNA_CONSTANTS.ATOM_NUM_MAX])), dim = 0)
        #print(all_atom_mask)
        #print(all_atom_mask.shape)#torch.float32

        gt_frames = Rigid.from_3_points(
            p_neg_x_axis=all_atom_positions[..., 0, :],
            origin=all_atom_positions[..., 1, :],
            p_xy_plane=all_atom_positions[..., 2, :],
            eps=1e-8,
        )

        #a1 = Rigid.to_tensor_4x4(gt_frames)
        #print(f'a1: {a1}')
        #print(f'a1: {a1.shape}') #torch.Size([100, 4, 4])
        gt_tensor_7 = gt_frames.to_tensor_7()
        #print(f'gt_tensor_7: {gt_tensor_7.shape}') 

        gt_mask = torch.ones([n_seq])
        gt_mask = torch.cat((gt_mask, torch.zeros([max_len-n_seq])), dim = 0)
        #print(gt_mask)
        #print(gt_mask.shape) 
     
        # tokens
        #print(config.input_a3m)#config.input_a3m=./example3/input/.fasta not used
        data_dict = get_features(pdb_id, seq, []) 

        #print('data_dict {}'.format( data_dict)) 
        #data_dict {'seq': 'GGGCAAGCCC', 'tokens': tensor([[[6, 6, 6, 7, 4, 4, 6, 7, 7, 7]]]), 'rna_fm_tokens': tensor([[6, 6, 6, 5, 4, 4, 6, 5, 5, 5]])}
        #A4 U5 G6 C7 tokens  
        #A4 C5 G6 U7 rna_fm_tokens'
        data['pdb'] = pdb_id
        data['frames'] = gt_tensor_7
        data['frames_mask'] = gt_mask
        data['all_atom_positions'] = all_atom_positions      
        data['all_atom_mask'] = all_atom_mask       
        data['seq'] = data_dict['seq']
        data['tokens'] = data_dict['tokens']
        data['rna_fm_tokens'] = data_dict['rna_fm_tokens']

        #print('data[tokens] {}'.format(data['tokens']))
        #print('data[rna_fm_tokens] {}'.format(data['rna_fm_tokens']))

        batch.append(data) # 

    return batch


class PDB_Manager(Dataset):  
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, index):
        data = self.batch[index]  # dictionary
        return data  



def main(config):
    '''
    testing pipeline
    '''
    start = time.time()
    os.makedirs(config.output_dir, exist_ok=True)

    logger = logging.getLogger('RhoFold testing')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(f'{config.output_dir}/log.txt', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f'Constructing RhoFold')
    #logger.info(f'    loading {config.ckpt}')

    if config.single_seq_pred:
        config.input_a3m = config.input_fas
        logger.info(f"Input_a3m is None, the modeling will run using single sequence only (input_fas)")

    elif config.input_a3m is None:
        config.input_a3m = f'{config.output_dir}/seq.a3m'
        databases = [f'{config.database_dpath}/rnacentral.fasta', f'{config.database_dpath}/nt']
        blast = BLASTN(binary_dpath=config.binary_dpath, databases=databases)
        blast.query(config.input_fas, f'{config.output_dir}/seq.a3m', logger)
        logger.info(f"Input_a3m {config.input_a3m}")

    else:
        logger.info(f"Input_a3m {config.input_a3m}")

    with timing('RhoFold testing', logger=logger):
        config.device = get_device(config.device)
        logger.info(f'    testing using device {config.device}')

        # input RNA sequence and groud truth 3d coordinates
        outfile = 'result.txt'
        max_len = rhofold_config.max_len_seq        
   
        infile = './example/rho2_coords_%s.pkl'%max_len  #10:6, 20:92, 30:216 
        with open(infile,"rb") as f:
            coord_atom = pickle.load(f)
        rho_coords = coord_atom['coords'] #constants.py　rho format revision
        atom_dict= coord_atom['atom_dict']
        #print(rho_coords)
        #print(atom_dict)

        batch = pdb_data_input(rho_coords, config, max_len)

        batch_size = rhofold_config.data_module.data_loaders.batch_size

        testing_set = PDB_Manager(batch[-4:])
        testing_generator = DataLoader(testing_set, **rhofold_config.data_module.data_loaders)

        model = RhoFold(rhofold_config)
        model.load_state_dict(torch.load(f'{config.output_dir}/esm1b_fape_rmsd_3.pt'))
        model = model.to(config.device)
        model.eval()      
               
        for batch_i, data in enumerate(testing_generator):
            print('idx: {}, data[frames]: {}'.format(batch_i, data['frames'].shape)) #torch.Size([B, L, 7])
            #print('idx, tokens {} {} '.format(batch_i, data['tokens'].shape)) #[B, L,20]
            #print('frames_mask {}'.format(data['frames_mask'].shape)) #[B, L]
            #print(data['tokens'].squeeze(1)*data['frames_mask'])

            output = model(tokens=data['tokens'].to(config.device),
                           rna_fm_tokens=data['rna_fm_tokens'].to(config.device),
                           frame_mask = data['frames_mask'].to(config.device),
                           all_atom_mask = data['all_atom_mask'].to(config.device),
                           )

            output['all_atom_pred_pos'] = output['cord_tns_pred'][0].reshape(-1,max_len,RNA_CONSTANTS.ATOM_NUM_MAX,3)

            loss_func = RhoFoldLoss(rhofold_config).to(config.device)
      
            cum_loss, losses, measures = loss_func(output, data['frames'].to(config.device), data['frames_mask'].to(config.device), data['all_atom_positions'].to(config.device), data['all_atom_mask'].to(config.device), (data['tokens'].squeeze(1)*data['frames_mask']).to(config.device) )


            print('cum_loss: {}'.format(cum_loss)) 
            print('fape: {}'.format(losses['fape']))
            print('rmsd: {}'.format(losses['rmsd']))
            print('loss plddt: {}'.format(losses['plddt_loss']))
            print('loss tm: {}'.format(losses['tm_loss']))

            print('lddt: {}'.format(measures['lddt']))
            print('plddt in loss: {}'.format(measures['plddt']))
            print('plddt {}'.format(output['plddt'][1].detach().cpu() )) 

            print('ptm in loss: {}'.format(output['ptm'])) 

            logger.info('batch_i: {}, fape: {}'.format(batch_i, losses['fape']))
            assert not torch.isnan(cum_loss).any()
            cum_losses=torch.norm(cum_loss)

        print("Execution time {}".format(time.time()-start))

        output_dict={}
        output_dict['cum_loss'] = cum_loss
        output_dict['fape'] = losses['fape']
        output_dict['rmsd'] = losses['rmsd']
        output_dict['loss plddt'] = losses['plddt_loss']
        output_dict['loss tm'] = losses['tm_loss']
        output_dict['lddt'] = measures['lddt']
        output_dict['plddt_'] = measures['plddt']
        output_dict['plddt'] = output['plddt'][1].detach().cpu()
        output_dict['ptm'] = output['ptm']

        with open(outfile +'.txt','w') as f:
                f.write('cum_loss: {} \n'.format(output_dict['cum_loss'])) 
                f.write('fape: {} \n'.format(output_dict['fape']))
                f.write('rmsd: {} \n'.format(output_dict['rmsd']))
                f.write('loss plddt: {} \n'.format(output_dict['loss plddt']))
                f.write('loss tm: {} \n'.format(output_dict['loss tm']))

                f.write('lddt: {} \n'.format(output_dict['lddt']))
                f.write('plddt in loss: {} \n'.format(output_dict['plddt_']))               
                f.write('plddt {} \n'.format(output_dict['plddt']))
                f.write('ptm {} \n'.format(output_dict['ptm']))

                f.write("Execution time {} \n".format(time.time()-start))

        with open(outfile +'.pkl','wb') as f:
            pickle.dump(output_dict, f)
                

        for idx in range(output['all_atom_pred_pos'].shape[0]):
            output_dir = f'{config.output_dir}/%s' % data['pdb'][idx]  
            os.makedirs(output_dir, exist_ok=True)

            #print(output['c4_'][0].shape) # torch.Size([B, 40, L, L])

            # Secondary structure, .ct format
            ss_prob_map = torch.sigmoid(output['ss'][idx, 0]).data.cpu().numpy()  #torch.Size([B, 1, L, L])
            ss_file = f'{output_dir}/ss.ct'
            save_ss2ct(ss_prob_map, data['seq'][idx], ss_file, threshold=0.5)

            # Dist prob map & Secondary structure prob map, .npz format
            npz_file = f'{output_dir}/results.npz'
            np.savez_compressed(npz_file,
                                dist_n = torch.softmax(output['n'][0][idx].squeeze(0), dim=0).data.cpu().numpy(),
                                dist_p = torch.softmax(output['p'][0][idx].squeeze(0), dim=0).data.cpu().numpy(),
                                dist_c = torch.softmax(output['c4_'][0][idx].squeeze(0), dim=0).data.cpu().numpy(),
                                ss_prob_map = ss_prob_map,
                                plddt = output['plddt'][0][idx].data.cpu().numpy(),
                                )

            # Save the prediction
            unrelaxed_model = f'{output_dir}/unrelaxed_model.pdb'

            # The last cords prediction
            node_cords_pred = output['all_atom_pred_pos'][idx].squeeze(0)
            #print(node_cords_pred.shape)
    
            model.structure_module.converter.export_pdb_file(data['seq'][idx] ,
                                                             node_cords_pred.data.cpu().numpy(),
                                                             path=unrelaxed_model, chain_id=None,
                                                             confidence=output['plddt'][0][idx].data.cpu().numpy(),
                                                             logger=logger)

            # Amber relaxation
            if config.relax_steps is not None and config.relax_steps > 0:
                with timing(f'Amber Relaxation : {config.relax_steps} iterations', logger=logger):
                    amber_relax = AmberRelaxation(max_iterations=config.relax_steps, logger=logger)
                    relaxed_model = f'{output_dir}/relaxed_{config.relax_steps}_model.pdb'
                    amber_relax.process(unrelaxed_model, relaxed_model)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="Default cpu. If GPUs are available, you can set --device cuda:<GPU_index> for faster prediction.", default='cpu')
    parser.add_argument("--ckpt", help="Path to the pretrained model, default ./pretrained/model.pt", default='./pretrained/model.pt')
    parser.add_argument("--input_fas", help="Path to the input fasta file. Valid nucleic acids in RNA sequence: A, U, G, C", required=True)
    parser.add_argument("--input_a3m", help="Path to the input msa file. Default None."
                                            "If --input_a3m is not given (set to None), MSA will be generated automatically. ", default=None)
    parser.add_argument("--output_dir", help="Path to the output dir. "
                                             "3D prediction is saved in .pdb format. "
                                             "Distogram prediction is saved in .npz format. "
                                             "Secondary structure prediction is save in .ct format. ", required=True)
    parser.add_argument("--relax_steps", help="Num of steps for structure refinement, default 1000.", default = 1000)
    parser.add_argument("--single_seq_pred", help="Default False. If --single_seq_pred is set to True, "
                                                       "the modeling will run using single sequence only (input_fas)", default=False)
    parser.add_argument("--database_dpath", help="Path to the pretrained model, default ./database", default='./database')
    parser.add_argument("--binary_dpath", help="Path to the pretrained model, default ./rhofold/data/bin", default='./rhofold/data/bin')

    args = parser.parse_args()
    main(args)
