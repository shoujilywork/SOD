#ifndef NOV_5_H
#define NOV_5_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;
VectorXd nov_5(const VectorXd& f, double h);
VectorXd nov_5_n(const VectorXd& f, double h);

#endif