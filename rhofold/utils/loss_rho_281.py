# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
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

from functools import partial
import logging
import ml_collections
import numpy as np
import torch
import torch.nn as nn

#from torch.distributions.bernoulli import Bernoulli
from typing import Dict, Optional, Tuple
from torch.autograd import Variable

from rhofold.utils import constants
#from openfold.utils import feats
from rhofold.utils.rigid_utils import Rotation, Rigid
from rhofold.utils.constants import RNA_CONSTANTS

from rhofold.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    masked_mean,
    permute_final_dims,
    batched_gather,
)

def softmax_cross_entropy(logits, labels):
    #print('logits {}'.format(logits.shape)) #torch.Size([B, L, 50]) 
    #print('labels {}'.format(labels.shape)) #torch.Size([B, L, 50])

    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    #print('labels * torch.nn.functional.log_softmax(logits, dim=-1) {}'.format(labels * torch.nn.functional.log_softmax(logits, dim=-1))) #torch.Size([B, L, 50])
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions  
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    #print(pred_frames.invert()[..., None].shape) # torch.Size([B, 100, 1])
    #print(pred_positions[..., None, :, :].shape) # torch.Size([B, 1, 100, 3])
    #print(local_pred_pos.shape) #torch.Size([3, 100, 100, 3])
    #print(local_pred_pos[0,:5,0,:]) #torch.Size([3, 100, 100, 3])
    #print(local_pred_pos[0,0,:5,:]) #torch.Size([3, 100, 100, 3])


    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    #print('length_scale {}'.format(length_scale)) #10.0

    normed_error = error_dist / length_scale
    #normed_error = error_dist
    #print('normed_error 1 {}'.format(normed_error.shape)) #torch.Size([B, 3, 100, 100])
    #print('frames_mask[..., None] {}'.format(frames_mask[..., None].shape)) # torch.Size([B, 3, 100, 1])
    normed_error = normed_error * frames_mask[..., None]
    #print('normed_error 2 {}'.format(normed_error.shape)) # torch.Size([B, 3, 100, 100])
    #print('positions_mask[..., None, :] {}'.format(positions_mask[..., None, :].shape)) #torch.Size([B, 3, 1, 100])
    normed_error = normed_error * positions_mask[..., None, :]
    #print('normed_error {}'.format(normed_error.shape)) # torch.Size([B, 3, 100, 100])

    #print('frames_mask {}'.format(frames_mask))
    #print('frames_mask {}'.format(frames_mask.shape)) #torch.Size([B, 3, 100])
    #print('positions_mask {}'.format(positions_mask.shape)) # torch.Size([B, 3, 100])
    

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    #print('normed_error 3  {}'.format(normed_error.shape)) # normed_error 3  torch.Size([B, 3, 100])
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    #print('normed_error 4  {}'.format(normed_error.shape)) #torch.Size([B, 3, 100])
    normed_error = torch.sum(normed_error, dim=-1)
    #print('normed_error 5  {}'.format(normed_error.shape)) #torch.Size([B, 3])
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    #print('normed_error 6  {}'.format(normed_error.shape)) #torch.Size([B, 3])

    return normed_error


def backbone_loss(
    traj: torch.Tensor,
    backbone_rigid_tensor: torch.Tensor, #ground truth
    backbone_rigid_mask: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:

    pred_aff = Rigid.from_tensor_7(traj)  # tensor_7 to Rigid
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.

    #gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)  #  WRONG
    gt_aff = Rigid.from_tensor_7(backbone_rigid_tensor)  #  tensor_7 to Rigid
    gt_aff = Rigid(
        Rotation(rot_mats=gt_aff.get_rots().get_rot_mats(), quats=None),
        gt_aff.get_trans(),
    )
    #print('pred_aff.get_trans(): {}'.format(pred_aff.get_trans()))
    #print('pred_aff.get_trans(): {}'.format(pred_aff.get_trans().shape)) #torch.Size([B, 100, 3])

    
    fape_loss = compute_fape(
        pred_aff,     #Rigid
        gt_aff[None], #Rigid
        backbone_rigid_mask[None],
        pred_aff.get_trans(),       #position
        gt_aff[None].get_trans(),   #position
        backbone_rigid_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )
    #print('fape_loss {}'.format(fape_loss)) #tensor([[0.9746, 0.9546, 0.9645]], device='cuda:0')
    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)
    #print('fape_loss {}'.format(fape_loss)) # 0.9645881056785583

    return fape_loss


def fape_loss(
    out: Dict[str, torch.Tensor],
    #batch: Dict[str, torch.Tensor],
    gt_tensor_7: torch.Tensor,
    gt_mask: torch.Tensor,
    config: ml_collections.ConfigDict,
    ) -> torch.Tensor:
    bb_loss = backbone_loss(
        #traj=out["sm"]["frames"],  #from openfold
        #**{**batch, **config.backbone},  
        out["frames"][-1],  #rhofold prediction
        gt_tensor_7, gt_mask,
    )

    #print('bb_loss {}'.format(bb_loss))

    loss = config.backbone.weight * bb_loss
    #loss = bb_loss 
    
    # Average over the batch dimension
    loss_2 = torch.mean(loss)

    return loss_2, loss


def compute_rmsd(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions  
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )  

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    ###normed_error = error_dist / length_scale
    normed_error = error_dist
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = ( normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None] ) #[1, B, L]
    #print('normed_error {}'.format(normed_error.shape))
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1)) #[1, B]
    #print('normed_error {}'.format(normed_error.shape))
    return normed_error**0.5


def all_atom_loss(
    traj: torch.Tensor,
    backbone_rigid_tensor: torch.Tensor, #ground truth
    backbone_rigid_mask: torch.Tensor,

    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,

    use_clamped_rmsd: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:

    pred_aff = Rigid.from_tensor_7(traj)  # tensor_7 to Rigid
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    gt_aff = Rigid.from_tensor_7(backbone_rigid_tensor)  #  tensor_7 to Rigid
    gt_aff = Rigid(
        Rotation(rot_mats=gt_aff.get_rots().get_rot_mats(), quats=None),
        gt_aff.get_trans(),
    )

    rmsd_loss = compute_rmsd(
        pred_aff,     #Rigid
        gt_aff[None], #Rigid
        backbone_rigid_mask[None],
        all_atom_pred_pos,    #position
        all_atom_positions,   #position
        all_atom_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    
    #print('rmsd {}'.format(rmsd_loss.shape)) #[B,8]

    # Average over the batch dimension
    # rmsd_loss = torch.mean(rmsd_loss)

    return rmsd_loss


def rmsd_loss(
    out: Dict[str, torch.Tensor],
    gt_tensor_7: torch.Tensor,
    gt_mask: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    config: ml_collections.ConfigDict,
    ) -> torch.Tensor:
    rmsd = all_atom_loss(
        #traj=out["sm"]["frames"],  #from openfold
        #**{**batch, **config.backbone},  
        out["frames"][-1],  #rhofold prediction
        gt_tensor_7, 
        gt_mask,
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
    )

    #loss = config.backbone.weight * rmsd
    loss = rmsd
    #print('config.backbone.weight {}'.format(config.backbone.weight))
    #print('rmsd {}'.format(rmsd))
    #print('loss {}'.format(loss))
    # Average over the batch dimension
    loss = torch.mean(loss)
    
    return loss, rmsd



def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    #print('dmat_true {}'.format(dmat_true))
    #print('dmat_pred {}'.format(dmat_pred))
    
    dists_to_score = (
        (dmat_true < cutoff) #true or false = 1 or 0
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )
    #print('dmat_true {}'.format(dmat_true.shape)) #torch.Size([B, 100, 100])
    #print('dmat_pred {}'.format(dmat_pred.shape)) #torch.Size([B, 100, 100])

    dist_l1 = torch.abs(dmat_true - dmat_pred)
    #print('dists_to_score {}'.format(dists_to_score))
    #print('dist_l1 {}'.format(dist_l1))
    #print('dist_l1 {}'.format(dist_l1.shape)) #torch.Size([B, 100, 100])

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)  # O or 1
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    #norm = 1.0 / (eps + torch.sum(all_atom_mask, dim=-2))
    #print(norm)
    
    score = norm * torch.sum(dists_to_score * score, dim=dims)

    #print('score {}'.format(score)) #score tensor([[0.1500, 0.1875, 0.0000, 0.1250, 0.0750, 0.0000, 0.0625,...],...[...  ]], device='cuda:0', dtype=torch.float64)

    return score


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    #print('plddt logits: {}'.format(logits)) #[B,20,50]
    #print('plddt logits: {}'.format(logits.shape)) #[B,20,50]
    
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    """
    print('bounds: {}'.format(bounds.shape)) #[50]
    print('probs: {}'.format(probs))
    print('probs: {}'.format(probs.shape)) #[B, 20, 50]
    print('probs.shape[:-1]: {}'.format(probs.shape[:-1])) #[B,20]
    print('bounds.view: {}'.format(((1,) * len(probs.shape[:-1])))) #(1,1)
    print('bounds.view: {}'.format(*((1,) * len(probs.shape[:-1])))) #1
    print('bounds.view: {}'.format(*bounds.shape)) #50
    print('bounds.view: {}'.format(bounds.view(B,50))) 
    print('probs*bounds: {}'.format(probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape)))
    """
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    #print('pred_lddt_ca: {}'.format(pred_lddt_ca)) #[B,20]
    #print('pred_lddt_ca: {}'.format(pred_lddt_ca.shape)) #[B,20]
    pred_lddt_ca = torch.mean(pred_lddt_ca, dim=(-2,-1))
    #print('pred_lddt_ca: {}'.format(pred_lddt_ca)) # scalar
    
    return pred_lddt_ca * 100


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    ca_pos = 1 #C1'
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :] # res_id, atom_id, 
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos, 
        all_atom_positions, #ground truth
        all_atom_mask, 
        cutoff=cutoff, 
        eps=eps
    )

    score = score.detach() # なぜdetachするのか
    #print('lddt score {}'.format(score))#[B,L]
    
    bin_index = torch.floor(score * no_bins).long()
    #print('bin_index {}'.format(bin_index)) # [B,L] [[ 5,  0,  0, 12,  7,  0,  7,  8, 16,  8,  0,  0, 16,
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    #print('bin_index {}'.format(bin_index)) 
    
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )
    #print('lddt_ca_one_hot {}'.format(lddt_ca_one_hot)) #torch.Size([B, L, 50])
    #print('logits {}'.format(logits) ) #[B, L, 50]
    #print('logits {}'.format(logits.shape) ) #[B, L, 50]

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot) 
    
    #print('errors {}'.format(errors)) #tensor([[3.9869, 4.0198, 3.8508, 3.8519, 3.9114,...]], device='cuda:0')
    #print('errors {}'.format(errors.shape)) #torch.Size([B, L])

    all_atom_mask = all_atom_mask.squeeze(-1)
    #print('all_atom_mask: {}'.format(all_atom_mask))

    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )
    #print('loss {}'.format(loss)) #tensor([2.3065]

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    ) 
    #print('loss {}'.format(loss)) #grad_fn=<MulBackward0>

    # Average over the batch dimension
    loss = torch.mean(loss)

    #lddt score
    #print('score*all_atom_mask: {}'.format(score*all_atom_mask))
    #print('torch.sum(all_atom_mask, dim=-1): {}'.format(torch.sum(all_atom_mask, dim=-1)))
    score = torch.sum(score*all_atom_mask, dim=-1) / (eps + torch.sum(all_atom_mask, dim=-1))
    #print('score {}'.format(score))
    score = torch.mean(score)
    #print('score {}'.format(score))

    plddt_val = compute_plddt(logits)

    return loss, score, plddt_val


def distogram_loss(
    logits,
    pseudo_beta, #ground truth
    pseudo_beta_mask,
    min_bin=2.3125, #from openfold for proteins should adjust to RNA
    max_bin=21.6875,
    no_bins=40, #64  
    eps=1e-6,
    **kwargs,
):
    
    #print('distogram logits: {}'.format(logits.shape)) #torch.Size([B, L, L, 40])
    #print('pseudo_beta: {}'.format(pseudo_beta.shape)) #torch.Size([B, 100, 3])
    #print('pseudo_beta_mask: {}'.format(pseudo_beta_mask.shape)) # torch.Size([B, 100])
   
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2
    
    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)
    #print('distogram logits: {}'.format(logits.shape)) #torch.Size([B, L, L, 40])
    #print('distogram torch.nn.functional.one_hot(true_bins, no_bins): {}'.format(torch.nn.functional.one_hot(true_bins, no_bins).shape)) #torch.Size([B, L, L, 40])

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

"""
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

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights
    
    print('weighted in compute_tm {}'.format(weighted))
    argmax = (weighted == torch.max(weighted)).nonzero()[0] #IndexError: index 0 is out of bounds for dimension 0 with size 0
    return per_alignment[tuple(argmax)]
"""

def tm_loss(
    logits,
    final_affine_tensor, # pred
    backbone_rigid_tensor, # gt
    backbone_rigid_mask,  # gt
    resolution,
    max_bin=31,
    no_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    **kwargs,
):
    pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_7(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]  # pred_aff
        return affine.invert()[..., None].apply(pts)

    # pred vs gt
    sq_diff = torch.sum(
        (_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1
    )

    sq_diff = sq_diff.detach()  # なぜdetachするのか

    #print('sq_diff: {}'.format(sq_diff))
    #print('sq_diff: {}'.format(sq_diff.shape)) #torch.Size([B, 100, 100])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )

    square_mask = (
        backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]
    )

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    # Average over the loss dimension
    loss = torch.mean(loss)

    #pred_tm = compute_tm(logits, **kwargs) #scalar, it does not use the ground truth
 
    return loss

res2idx = {'A': 4, 'C': 5, 'G':6, 'U':7, '-':8 }
res2O3_ = {4:7, 5:7, 6:7, 7:7}
          # A    C     G     U 
res2O5_ = {4:18, 5:16, 6:19, 7:16}
res2P   = {4:19, 5:17, 6:20, 7:17}
res2OP1 = {4:20, 5:18, 6:21, 7:18}
res2OP2 = {4:21, 5:19, 6:22, 7:19}

def length(a, b):
    return torch.norm(a-b, dim=-1)

#from RoseTTA
# ideal N-C distance, ideal cos(CA-C-N angle), ideal cos(C-N-CA angle)
# for NA, we do not compute this as it is not computable from the stubs alone
def calc_BB_bond_geom(
    seq,  pred, mask, 
    ideal_OP=1.607, ideal_OPO=-0.3106, ideal_OPC=-0.4970, 
    sig_len=0.02, sig_ang=0.05,  eps=1e-6, **config):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''

    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        #print(f'ABn : {ABn}')
        #print(f'AB*BC : {AB*BC}')
        #print(f'AB*BC : {(AB*BC).shape}')
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    #print(pred)
    #print('seq: {}'.format(seq)) #[B,L]
    
    B, L = pred.shape[:2]
    #print(f'B, L: {B} {L}')
    mask = mask[:,1:]

    # bond length: P-O
    blen_OP_pred  = length(pred[:,:-1,8], pred[:,1:,4]).reshape(B,L-1) # (B, L-1)    C3', P
    OP_loss = torch.clamp( torch.abs(blen_OP_pred - ideal_OP) - sig_len, min=0.0 ) * mask

    #print('OP_loss {}'.format(OP_loss))
    #print('torch.sum(mask, dim=1) {}'.format(torch.sum(mask, dim=-1)))

    OP_loss = OP_loss / torch.sum(mask, dim=1).unsqueeze(0).T
    blen_loss = OP_loss

    # bond angle: O1PO, O2PO, OPC
    bang_O1PO_pred = cosangle(pred[:,:-1,8], pred[:,1:,4], pred[:,1:,5]).reshape(B,L-1)  #C3', P, OP1
    bang_O2PO_pred = cosangle(pred[:,:-1,8], pred[:,1:,4], pred[:,1:,6]).reshape(B,L-1)  #C3', P, OP2
    O1PO_loss = torch.clamp( torch.abs(bang_O1PO_pred - ideal_OPO) - sig_ang,  min=0.0 )
    O2PO_loss = torch.clamp( torch.abs(bang_O2PO_pred - ideal_OPO) - sig_ang,  min=0.0 )

    #print('O1PO_loss {}'.format(O1PO_loss))
    #print('O2PO_loss {}'.format(O2PO_loss))
    O1PO_loss = O1PO_loss / torch.sum(mask, dim=1).unsqueeze(0).T    
    O2PO_loss = O2PO_loss / torch.sum(mask, dim=1).unsqueeze(0).T

    bang_loss = O1PO_loss + O2PO_loss

    return  torch.mean(blen_loss) + torch.mean(bang_loss)


#from RoseTTA
def calc_clash(xs, mask):
    DISTCUT=2.0 # (d_lit - tau) from AF2 MS
    B, L = xs.shape[:2] # sequence length
    #print('B, L: {} {}'.format(B, L))
    dij = torch.sqrt(
        torch.sum( torch.square( xs[...,None,:]-xs[...,None,:,:] ), dim=-1 ) + 1e-8  #batch, length, coord
    )
    #print('dij: {}'.format(dij)) #torch.Size([1, 1081, 1081])
    #print(dij.shape)

    allmask = mask[:,:,None]*mask[:,None,:]
    allmask[:,torch.arange(L),torch.arange(L)] = 0 # res-self

    #print('allmask {}'.format(allmask))
    #print('DISTCUT-dij {}'.format(DISTCUT-dij))
    #print('(DISTCUT-dij)*allmask {}'.format((DISTCUT-dij)*allmask)) 
    #print('(DISTCUT-dij)*allmask {}'.format((dij*allmask).shape)) 
    
    #print(torch.clamp((DISTCUT-dij)*allmask,0.0)) 
    #print(torch.sum(mask))
    clash = torch.sum( torch.clamp((DISTCUT-dij)*allmask, 0.0) ) / (torch.sum(mask) * B)
    #print(clash)

    return clash

 

class RhoFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""
    def __init__(self, config):
        super(RhoFoldLoss, self).__init__()
        self.config = config

    def forward(self, out, gt_tensor_7, gt_mask, all_atom_positions, all_atom_mask, seq, _return_breakdown=False):

        #print(out['frames'][-1].shape) #torch.Size([B, L, 7])
        #print(gt_tensor_7.shape) #torch.Size([B, L, 7])
        max_len = all_atom_mask.shape[-2]
        #print('max_len {}'.format(max_len))

        loss_fns = {
            "fape": lambda: fape_loss(
                out, # Prediction, Dict
                gt_tensor_7, gt_mask,
                self.config.fape,
            ),

            "rmsd": lambda: rmsd_loss(
                out, # Prediction, Dict
                gt_tensor_7, 
                gt_mask,
                out["all_atom_pred_pos"].reshape(-1, max_len*RNA_CONSTANTS.ATOM_NUM_MAX, 3), 
                all_atom_positions.reshape(-1, max_len*RNA_CONSTANTS.ATOM_NUM_MAX, 3), 
                all_atom_mask.reshape(-1, max_len*RNA_CONSTANTS.ATOM_NUM_MAX),
                self.config.rmsd,
            ),        
            "plddt_loss": lambda: lddt_loss(
                logits = out["plddt"][2],
                all_atom_pred_pos = out["all_atom_pred_pos"],
                all_atom_positions = all_atom_positions, 
                all_atom_mask = all_atom_mask, 
                resolution = 1, ### temporary setting we need to set it
                **self.config.plddt_loss,
            ),

            "distogram_loss": lambda: distogram_loss(
                logits=out["c4_"].permute(0,2,3,1), # torch.Size([B, 40, 20, 20]) p c4'(0) n
                pseudo_beta = all_atom_positions[..., 0, :], #0:C4' 1:C1'
                pseudo_beta_mask = all_atom_mask[..., 0 ], # pseudo_beta_mask = all_atom_mask[..., 1 : (1+1)]
                **self.config.distogram,
            ),
  
            "tm_loss" : lambda: tm_loss(
                logits=out["tm"], 
                final_affine_tensor = out['frames'][-1],
                backbone_rigid_tensor = gt_tensor_7,
                backbone_rigid_mask = gt_mask,  # gt
                resolution = 1, ### temporary setting
                **self.config.tm,
            ),
            "clash" : lambda: calc_clash(
                out["all_atom_pred_pos"].reshape(-1, max_len*RNA_CONSTANTS.ATOM_NUM_MAX, 3), 
                all_atom_mask.reshape(-1, max_len*RNA_CONSTANTS.ATOM_NUM_MAX),
            ),               
            "binding_loss" : lambda: calc_BB_bond_geom(
                seq,
                out["all_atom_pred_pos"], #seq=tokens [B, L, 23, 3]
                gt_mask,
                **self.config,
            ),  
        }

        
        cum_loss = 0 #torch.tensor(0., requires_grad=True)
        losses = {}
        measures = {}
        for loss_name, loss_fn in loss_fns.items():

            if loss_name in ["fape", "plddt_loss", "rmsd", "tm_loss" ]: #, "rmsd", "distogram_loss", "clash", ]: # 
                if loss_name == "fape":
                    weight = 1.0
                    loss, _ = loss_fn()
                elif loss_name == "plddt_loss":
                    weight = 0.1*0.01
                    loss, lddt_val, plddt_val = loss_fn()                   
                elif loss_name == 'rmsd':
                    weight = 0.1
                    loss, _ = loss_fn()
                elif loss_name == 'distogram_loss':
                    weight = 0.1
                    loss = loss_fn()
                elif loss_name == 'tm_loss':
                    weight = 0.1
                    loss = loss_fn()         
                elif loss_name == 'clash':
                    weight = 0.1
                    loss = loss_fn()
                elif loss_name == 'binding_loss':
                    weight = 0.1
                    loss = loss_fn() # blen_PO3_loss & bang_O3PO5_loss are good measures.
                    assert not torch.isnan(loss).any()
                else:
                    pass
                
                #print('cum_loss 01 {}'.format(loss)) #tensor(3.6356, device='cuda:0', dtype=torch.float64)
                #print(loss) #tensor(0.4791, device='cuda:0')

                if(torch.isnan(loss) or torch.isinf(loss)):
                    #for k,v in batch.items():
                    #    if(torch.any(torch.isnan(v)) or torch.any(torch.isinf(v))):
                    #        logging.warning(f"{k}: is nan")
                    #logging.warning(f"{loss_name}: {loss}")
                    logging.warning(f"{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0., requires_grad=True)
                assert not torch.isnan(loss).any()
                cum_loss = cum_loss + weight * loss

                #print('cum_loss 1 {}'.format(cum_loss)) #tensor(3.6356, device='cuda:0', dtype=torch.float64)
                losses[loss_name] = loss.detach().clone()

                if loss_name in ["plddt_loss"] :
                    measures["plddt"] = plddt_val.detach().clone()  #duplication output['plddt'] in rhofold.py
                    measures["lddt"]  = lddt_val.detach().clone()

                else:
                    pass
        #losses["unscaled_loss"] = cum_loss.detach().clone()
        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        #seq_len = torch.mean(batch["seq_length"].float())
        #crop_len = batch["aatype"].shape[-1]
        #cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))
        #cum_loss = cum_loss * torch.sqrt(torch.tensor(gt_tensor_7.shape[1]))

        #print(f'cum_loss= {cum_loss}')
        #print(f"cum_loss.grad={cum_loss.grad}")

        losses["loss"] = cum_loss.detach().clone()
        #print('cum_loss 2 {}'.format(cum_loss)) #tensor(17.1943, device='cuda:0', dtype=torch.float64)

        return cum_loss, losses, measures


