// Exercise the real C++ writer with empty, single, and multi-candidate snapshots.
#include <epic_planner/decision_recorder.h>
#include <cassert>
#include <fstream>
int main(int argc, char** argv) {
  ros::init(argc, argv, "dst_recorder_probe");
  if (argc != 2) return 2;
  ros::NodeHandle nh("~");
  const std::string filename = argv[1];
  nh.setParam("collection/output_file", filename);
  auto graph = std::make_shared<TopoGraph>();
  auto node = [](float x) {
    auto n = std::make_shared<TopoNode>();
    n->center_ = Eigen::Vector3f(x, 0, 2); n->yaw_ = 0;
    return n;
  };
  graph->odom_node_ = node(0);
  auto a = node(1), b = node(2);
  a->is_viewpoint_ = b->is_viewpoint_ = true;
  graph->odom_node_->neighbors_.insert(a);
  graph->odom_node_->weight_[a] = 1;
  a->neighbors_.insert(b); a->weight_[b] = 1;
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(3,3);
  mat(0,1)=1;mat(0,2)=2;mat(1,2)=mat(2,1)=1;
  {
    fast_planner::DecisionRecorder writer(nh);
    auto position = graph->odom_node_->center_.cast<double>().eval();
    auto velocity = Eigen::Vector3d::Zero().eval();
    writer.record(graph,position,velocity,0,{}, {},graph->odom_node_->center_,false,{}, {},"no_candidates");
    writer.record(graph,position,velocity,0,{}, {},graph->odom_node_->center_,false,{}, {},"no_reachable_candidates");
    writer.record(graph,position,velocity,0,{a},{1},graph->odom_node_->center_,false,{}, {0,1},"single_candidate");
    writer.record(graph,position,velocity,0,{b,a},{2,1},a->center_,true,mat,{0,2,1},"lkh");
    assert(graph->odom_node_->neighbors_.size()==1);
    assert(graph->odom_node_->weight_.at(a)==1);
    assert(a->neighbors_.count(b)==1);
    assert(b->center_.x()==2);
  }
  // Reopening the same output must not overwrite it.
  {
    fast_planner::DecisionRecorder duplicate(nh);
  }
  ros::shutdown();
  return 0;
}
