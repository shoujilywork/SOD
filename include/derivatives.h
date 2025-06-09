#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;

VectorXd derivative_p(const VectorXd& u, double h);
VectorXd derivative_n(const VectorXd& u, double h);

#endif