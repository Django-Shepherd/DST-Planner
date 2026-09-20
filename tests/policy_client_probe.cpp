#include <epic_planner/policy_client.h>
#include <iostream>
int main(int argc,char**argv) {
  if(argc!=2)return 2;
  Eigen::MatrixXi reach=Eigen::MatrixXi::Ones(4,4);reach.diagonal().setZero();
  try {auto route=fast_planner::requestPolicy(argv[1],"{\"decision_id\":7}\n",7,reach,150);
    for(int x:route)std::cout<<x<<' ';return 0;
  } catch(const std::exception& e) {std::cerr<<e.what();return 1;}
}
