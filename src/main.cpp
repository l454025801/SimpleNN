#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::ArrayXXd;

int main()
{
  int s = 1;
  std::cout << -s << std::endl;
  //MatrixXd m(2,2);
  //MatrixXd a;
  //a = m+1;
  //std::cout << a << std::endl;
  ArrayXXd n(2,2);
  n(0,0) = 1;
  n(0,1) = 2;
  n(1,0) = 3;
  n(1,1) = 4;
  std::cout << n << std::endl;
  ArrayXXd a(2,2);
  a = n+1;
  std::cout << a << std::endl;
  std::cout << 1/(1+exp(-a.array())) << std::endl;
  //min(n.array(),a.array())
  MatrixXd b = MatrixXd::Random(3,3)*10;
  std::cout << b << std::endl;
}