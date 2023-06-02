# RNA3D
## Title
Training program construction (Python) for de novo RNA 3D structure prediction.
This is derived from E2Efold-3D (Rhofold) and OpenFold.
We constructed a training program by designing loss functions.

## Environment in Linux
See the envs diretory

## Run program
Users set parameters (input file name of sequences and their 3D coordinates, outfile name, max epoch, max len (max sequence length), batch size, use of RNA language model (True or False)) in "training_342.py"
If users use RNA-FM languate model, they install it from "https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth"
To calculate 3D coordinate
>sh main_1.sh

To display the training process
>python loss_dict.py

## References
E2Efold-3D: End-to-End Deep Learning Method for accurate denovo RNA 3D Structure Prediction.
arXiv:2207.01586v1, 4 Jul 2022

OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization.
bioRxiv: https://doi.org/10.1101/2022.11.20.517210, 22 Nov, 2022

### Our private ID
kurata34/myproject/py8/rhofold_k1
