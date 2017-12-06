//
//  LBFRegressor.h
//  my LBFRegressor for android
//
//  Created by zeng on 12/01/17.
//  Copyright (c) 2015 zeng. All rights reserved.
//


#ifndef __LBFRegressor_H_
#define __LBFRegressor_H_

#include "LBF.h"



class TreeNode {
public:
    //data
    bool issplit;
    int pnode;
    int depth;
    int cnodes[2];
    bool isleafnode;
    double thresh;
    double feat[4];
    std::vector<int> ind_samples;
   
    //Constructors
    TreeNode(){
        ind_samples.clear();
        issplit = 0;
        pnode = 0;
        depth = 0;
        cnodes[0] = 0;
        cnodes[1] = 0;
        isleafnode = 0;
        thresh = 0;
        feat[0] = 0;
        feat[1] = 0;
        feat[2] = 0;
        feat[3] = 0;
    }
    void Write(std::ofstream& fout){
        fout << issplit<<" "<< pnode <<" "<<depth<<" " << cnodes[0]<<" "<<cnodes[1]<<" "<<isleafnode<<" "
        << thresh<<" "<<feat[0]<<" "<<feat[1]<<" "<<feat[2]<<" "<<feat[3]<<std::endl;
    }
    void Read(std::ifstream& fin){
        fin >> issplit >> pnode >> depth >> cnodes[0] >> cnodes[1] >> isleafnode
        >> thresh >> feat[0] >> feat[1] >> feat[2] >> feat[3];
    }
};


class Tree{
public:

    // id of the landmark
    int landmarkID_;
    // depth of the tree:
    int max_depth_;
    // number of maximum nodes:
    int max_numnodes_;
    //number of leaf nodes and nodes
    int num_leafnodes_;
    int num_nodes_;

    // sample pixel featurs' number, use when training RF
    int max_numfeats_;
    double max_radio_radius_;
    double overlap_ration_;

    // leafnodes id
    std::vector<int> id_leafnodes_;
    // tree nodes
    std::vector<TreeNode> nodes_;

    Tree(){
    }

    void Init(Params global_params);
    //Predict
    //void Predict();

    // Read/ write
    void Read(std::ifstream& fin);
    void Write(std:: ofstream& fout);

};

class RandomForest{
public:
    std::vector<std::vector<Tree> > rfs_;
    int max_numtrees_;
    int num_landmark_;
    int max_depth_;
    int stages_;
    double overlap_ratio_;


    RandomForest(){

    }

    void Init(Params global_params);

    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
};

class LBFRegressor{
public:
    std::vector<RandomForest> RandomForest_;
    std::vector<std::vector<struct model*> > Models_;
    cv::Mat_<double> mean_shape_;
    std::vector<cv::Mat_<double> > shapes_residual_;
    int max_numstage_;

    Params global_params_;
public:
    LBFRegressor(){

    }
    ~LBFRegressor(){
        for(int i=0; i<Models_.size();i++) {
            for(int j=0; j<Models_[i].size(); j++) {
                delete Models_[i][j];
            }
        }
    }

    struct feature_node ** DeriveBinaryFeat(const RandomForest& randf,
                                            const std::vector<cv::Mat_<uchar> >& images,
                                            const std::vector<cv::Mat_<double> >& current_shapes,
                                            const std::vector<BoundingBox> & bounding_boxs);
    void GetCodefromRandomForest(struct feature_node *binfeature,
                                 const int index,
                                 const std::vector<Tree>& rand_forest,
                                 const cv::Mat_<uchar>& image,
                                 const cv::Mat_<double>& shape,
                                 const BoundingBox& bounding_box,
                                 const cv::Mat_<double>& rotation,
                                 const double scale);

    void GlobalPrediction(struct feature_node**,
                          std::vector<cv::Mat_<double> >& current_shapes,
                          const std::vector<BoundingBox> & bounding_boxs,
                          int stage);

    void ReleaseFeatureSpace(struct feature_node ** binfeatures,
                             int num_train_sample);

    cv::Mat_<double>  Predict(const cv::Mat_<uchar>& image,
                              const BoundingBox& bounding_box,
                              int initial_num);
    void Save(std::string modelPath, std::string regressorPath);
    void Load(std::string modelPath, std::string regressorPath);

    void WriteGlobalParam(std::ofstream& fout);
    void ReadGlobalParam(std::ifstream& fin);
    void WriteRegressor(std::ofstream& fout);
    void ReadRegressor(std::ifstream& fin);

};


#endif