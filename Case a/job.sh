#!/bin/bash
#SBATCH --account=def-oforion
#SBATCH --nodes=2                     # Request 1 node
#SBATCH --ntasks-per-node=64         # Total number of tasks per node
#SBATCH --mem=245G                    # Total memory per node
#SBATCH --time=7-00:00                # Time limit days-hours:minutes

# Load the required modules
module load StdEnv/2020
module load gcc/9.3.0
module load hdf5-mpi/1.10.6
module load boost/1.72.0
module load eigen
module load python/3.10.2
module load scipy-stack/2023b
module load mpi4py/3.0.3
module load petsc/3.17.1
module load slepc/3.17.2
module load scotch/6.0.9
module load fftw-mpi/3.3.8
module load ipp/2020.1.217
module load swig
module load flexiblas

# Activate FEniCS virtual environment and run your Python script
source /home/pasha77/fenics/bin/activate
source /home/pasha77/fenics/share/dolfin/dolfin.conf

srun python Isothermal_wr.py

# srun python Isothermal.py





