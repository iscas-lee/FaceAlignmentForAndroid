//
//  RandomForest.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBFRegressor.h"
using namespace std;
using namespace cv;

void RandomForest::Init(Params global_params) {
    max_numtrees_ = global_params.max_numtrees;
    num_landmark_ = global_params.landmark_num;
    max_depth_    = global_params.max_depth;
    overlap_ratio_ = global_params.bagging_overlap;
    
    // resize the trees
    rfs_.resize(num_landmark_);
    for (int i=0;i<num_landmark_;i++){
        rfs_[i].resize(max_numtrees_);
        for(int j=0; j<max_numtrees_; j++) {
            rfs_[i][j].Init(global_params);
        }
    }
}


void RandomForest::Write(std::ofstream& fout){
    fout << stages_ <<endl;
    fout << max_numtrees_<<endl;
    fout << num_landmark_<<endl;
    fout << max_depth_ <<endl;
    fout << overlap_ratio_ <<endl;
    for (int i=0; i< num_landmark_;i++){
        for (int j = 0; j < max_numtrees_; j++){
            rfs_[i][j].Write(fout);
        }
    }
}
void RandomForest::Read(std::ifstream& fin){
    fin >> stages_;
    fin >> max_numtrees_;
    fin >> num_landmark_;
    fin >> max_depth_;
    fin >> overlap_ratio_;
    for (int i=0; i< num_landmark_;i++){
        for (int j = 0; j < max_numtrees_; j++){
            rfs_[i][j].Read(fin);
        }
    }
}
