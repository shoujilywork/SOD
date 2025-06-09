#include "../include/derivatives.h"
#include "../include/nov_5.h"

VectorXd derivative_p(const VectorXd& u, double h) {
    return nov_5(u, h); 
}

VectorXd derivative_n(const VectorXd& u, double h) {
    return nov_5_n(u, h);
}