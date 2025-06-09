# Sod Shock Tube Solver in C++

This project implements a numerical solver for the **Sod shock tube problem**, a standard benchmark in computational fluid dynamics (CFD), using C++ with support from the Eigen library and OpenMP for parallelism.

## 📌 Overview

The Sod problem models a 1D in space and 4D in physical properties Riemann problem with a discontinuity in pressure and density, producing shock waves, rarefaction waves, and contact discontinuities.
This solver numerically integrates the 1D Euler equations using a finite difference scheme.

X direction is splitted into X-positive and X-negative
Physical dimensions are rho, p, E, and u.

6 cores are used to calculate derivatives.

The code is designed to be:

- Lightweight (header-only Eigen dependency)
- Parallelized using OpenMP for computing spatial derivatives
- Configurable for grid size, time step, and domain

## 🚀 Features

- Solves the 1D Euler equations (conservation of mass, momentum, and energy)
- Initial conditions based on Sod's classical setup
- Thomas algorithm support for tridiagonal matrix inversion (if used in implicit schemes)
- OpenMP-powered parallel derivative computations across 6 cores
- Adjustable parameters: spatial resolution, CFL condition, total time

## 🔧 Build Instructions

### Requirements

- C++17 or newer
- [Eigen](https://eigen.tuxfamily.org) (header-only linear algebra library)
- g++ with OpenMP support

### Compile (on Ubuntu/Linux)

```bash
g++ -std=c++17 -fopenmp -O2 main_omp.cpp -o sod_solver
