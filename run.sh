#!/bin/bash -l
#SBATCH --chdir /scratch/ghesquie/run/
#SBATCH --job-name Frag2D
#
#SBATCH -o out_Frag2D.%j
#SBATCH -e err_Frag2D.%j
#
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --nodes 1
#SBATCH --time 72:00:00
#SBATCH --mem 6000

module restore akantu_modules

echo STARTING AT $(date)

source /home/ghesquie/projects/lsms_codes/Frag2D/setupjed.sh

#python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 5e3 -t 0.8 -msh plate_0.01x0.01_npz3_P1.msh -mat AD995_cohesive_contact_m10_stable.dat
#python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 5e3 -t 0.6 -msh plate_0.01x0.01_npz3_P1.msh -mat AD995_cohesive_contact_m10_stable.dat
#python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 5e3 -t 0.4 -msh plate_0.01x0.01_npz5_P1.msh -mat AD995_cohesive_contact_m10_stable.dat
#python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 2.62e4 -t 0.4 -msh plate_0.01x0.01_npz4_P1.msh -mat AD995_cohesive_contact_u10_stable.dat
#python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 2.62e4 -t 0.4 -msh plate_0.01x0.01_npz5_P1.msh -mat AD995_cohesive_contact_u10.dat
#python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 5e3 -t 0.2 -msh plate_0.01x0.01_npz3_P1.msh -mat AD995_cohesive_contact_m10_stable.dat
python3 /home/ghesquie/projects/lsms_codes/Frag2D/src/main.py -sr 7e3 -t 0.1 -msh hollow_sphere_rin4.7e-03_rout5.0e-03_lc8.3e-05_p2.msh -mat material_linear_nsn.dat




echo FINISHED AT $(date)
