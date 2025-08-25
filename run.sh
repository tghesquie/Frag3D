#!/bin/bash -l
#SBATCH --chdir /scratch/ghesquie/Frag3D/
#SBATCH --job-name Frag3D
#SBATCH -o out_Frag3D.%j
#SBATCH -e err_Frag3D.%j
#SBATCH --account lsms-clip
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 3
#SBATCH --nodes 1
#SBATCH --time 172:00:00
#SBATCH --mem 21000
#SBATCH --qos bigmem

module purge
module restore akantu_env_py311

echo STARTING AT $(date)

source /home/ghesquie/projects/lsms_codes/Frag3D/setupjed.sh

python3 /home/ghesquie/projects/lsms_codes/Frag3D/src/main.py -v 50 -t 0.2 -msh plate_1.00e-02x1.00e-02x7.40e-05_npz3_P2.msh -mat material_linear_nsn.dat

echo FINISHED AT $(date)
