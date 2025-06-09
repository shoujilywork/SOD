#ifndef SPLIT_H
#define SPLIT_H

#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
split(const Eigen::VectorXd& r, const Eigen::VectorXd& p, const Eigen::VectorXd& u);

#endif