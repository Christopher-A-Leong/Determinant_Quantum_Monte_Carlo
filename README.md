# Determinant Quantum Monte Carlo

Implementation of the determinant quantum Monte Carlo algorithm, originally proposed by Blankenbecler, Scalapino, and Sugar (see reference [1]). 

## Overview

A powerful quantum Monte Carlo algorithm for computing properties of quantum many-body systems at finite-temperature is the determinant quantum Monte Carlo (DQMC) algorithm. Following from Trotter-Suzuki decompositions and the Blankenbecler-Scalapino-Sugar trace identity, the problem boils down to computing the Green's function. The Green's function, due to the decomposition of the interacting term (for instance through a Hubbard-Stratonovich decomposition) depends on an auxiliary field, on which the Monte Carlo sampling is performed. A full stable computation of the Green's function is computed following from the graded-QR decomposition procedure described in reference [2], meanwhile fast-updates after trial moves allow for three different methods. First is the usual Sherman-Morrison low-rank update algorithm. For better performance the delayed-update or submatrix-update algorithms, described in references [3] and [4] are alternatively used. 

## Requirements

See environment.yml file in the repository to create the conda environment related to this project.

## References

[1] R. Blankenbecler, D.J. Scalapino, and R.L. Sugar, Phys. Rev. D 24, 2278 (1981).

[2] A. Tomas, C.C. Chang, R. Scalettar, and Z. Bai, 2012 IEEE 26th International Parallel and Distributed Processing Symposium 308-319 (2012)

[3] F. Sun and X. Xu, Phys. Rev. B 109, 235140 (2024).

[4] F. Sun and X. Xu, SciPost Phys. 18, 055 (2025).

[5] F. Assaad, H. Evertz, Lecture Notes in Physics (Springer), Vol. 739, pp.277-356. 
