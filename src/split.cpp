#include "../include/split.h"

const double GAMMA = 1.4;

std::tuple<VectorXd, VectorXd, VectorXd, VectorXd, VectorXd, VectorXd> 
split(const VectorXd& r, const VectorXd& p, const VectorXd& u) {
    VectorXd ss = (GAMMA * p.cwiseQuotient(r)).cwiseSqrt();
    
    // Positive direction
    VectorXd A1 = (u + u.cwiseAbs()) / 2.0;
    VectorXd A2 = (u + ss + (u + ss).cwiseAbs()) / 2.0;
    VectorXd A3 = (u - ss + (u - ss).cwiseAbs()) / 2.0;
    
    VectorXd ep1 = (2.0 * (GAMMA - 1.0) * A1 + A2 + A3).cwiseProduct(r) / (2.0 * GAMMA);
    
    VectorXd temp = 2.0 * (GAMMA - 1.0) * A1.cwiseProduct(u) + A2.cwiseProduct(u + ss) + A3.cwiseProduct(u - ss);
    VectorXd ep2 = temp.cwiseProduct(r) / (2.0 * GAMMA);
    
    temp = (GAMMA - 1.0) * A1.cwiseProduct(u.cwiseProduct(u)) + A2.cwiseProduct((u + ss).cwiseProduct(u + ss)) / 2.0;
    VectorXd temp2 = A3.cwiseProduct((u - ss).cwiseProduct(u - ss)) / 2.0 + 
                    (3.0 - GAMMA) * (A2 + A3).cwiseProduct(ss.cwiseProduct(ss)) / (2.0 * (GAMMA - 1.0));
    VectorXd ep3 = (temp + temp2).cwiseProduct(r) / (2.0 * GAMMA);
    
    // Negative direction
    A1 = (u - u.cwiseAbs()) / 2.0;
    A2 = (u + ss - (u + ss).cwiseAbs()) / 2.0;
    A3 = (u - ss - (u - ss).cwiseAbs()) / 2.0;
    
    VectorXd en1 = (2.0 * (GAMMA - 1.0) * A1 + A2 + A3).cwiseProduct(r) / (2.0 * GAMMA);
    
    temp = 2.0 * (GAMMA - 1.0) * A1.cwiseProduct(u) + A2.cwiseProduct(u + ss) + A3.cwiseProduct(u - ss);
    VectorXd en2 = temp.cwiseProduct(r) / (2.0 * GAMMA);
    
    temp = (GAMMA - 1.0) * A1.cwiseProduct(u.cwiseProduct(u)) + A2.cwiseProduct((u + ss).cwiseProduct(u + ss)) / 2.0;
    temp2 = A3.cwiseProduct((u - ss).cwiseProduct(u - ss)) / 2.0 + 
           (3.0 - GAMMA) * (A2 + A3).cwiseProduct(ss.cwiseProduct(ss)) / (2.0 * (GAMMA - 1.0));
    VectorXd en3 = (temp + temp2).cwiseProduct(r) / (2.0 * GAMMA);

    return std::make_tuple(ep1, ep2, ep3, en1, en2, en3);

}