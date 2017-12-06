//
//  Tree.cpp
//  myopencv
//
//  Created by lequan on 1/23/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBFRegressor.h"
using namespace std;
using namespace cv;

inline double calculate_var(const vector<double>& v_1 ){
    if (v_1.size() == 0)
        return 0;
    Mat_<double> v1(v_1);
    double mean_1 = mean(v1)[0];
    double mean_2 = mean(v1.mul(v1))[0];
    return mean_2 - mean_1*mean_1;
    
}
inline double calculate_var(const Mat_<double>& v1){
    // calculate var
    double mean_1 = mean(v1)[0];
    double mean_2 = mean(v1.mul(v1))[0];
    return mean_2 - mean_1*mean_1;
    
}

void Tree::Init(Params global_params) { 
    overlap_ration_ = global_params.bagging_overlap;
    max_depth_ = global_params.max_depth;
    max_numnodes_ = pow(2, max_depth_)-1;
    nodes_.resize(max_numnodes_);
}

void Tree::Write(std:: ofstream& fout){
    fout << landmarkID_<<endl;
    fout << max_depth_<<endl;
    fout << max_numnodes_<<endl;
    fout << num_leafnodes_<<endl;
    fout << num_nodes_<<endl;
    fout << max_numfeats_<<endl;
    fout << max_radio_radius_<<endl;
   // fout << overlap_ration_ << endl;
    fout << 0.4 << endl;
    
    fout << id_leafnodes_.size()<<endl;
    for (int i=0;i<id_leafnodes_.size();i++){
        fout << id_leafnodes_[i]<< " ";
    }
    fout <<endl;
    
    for (int i=0; i <max_numnodes_;i++){
        nodes_[i].Write(fout);
    }
}
void Tree::Read(std::ifstream& fin){
    fin >> landmarkID_;
    fin >> max_depth_;
    fin >> max_numnodes_;
    fin >> num_leafnodes_;
    fin >> num_nodes_;
    fin >> max_numfeats_;
    fin >> max_radio_radius_;
    fin >> overlap_ration_;
    int num ;
    fin >> num;
    id_leafnodes_.resize(num);
    for (int i=0;i<num;i++){
        fin >> id_leafnodes_[i];
    }
    
    for (int i=0; i <max_numnodes_;i++){
        nodes_[i].Read(fin);
    }
}


