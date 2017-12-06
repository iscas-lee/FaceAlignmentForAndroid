//
//  LBFRegressor.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBFRegressor.h"
using namespace std;
using namespace cv;

struct feature_node ** LBFRegressor::DeriveBinaryFeat(
                                    const RandomForest& randf,
                                    const vector<Mat_<uchar> >& images,
                                    const vector<Mat_<double> >& current_shapes,
                                    const vector<BoundingBox> & bounding_boxs){
    
    // initilaize the memory for binfeatures
    struct feature_node **binfeatures;
    binfeatures = new struct feature_node* [images.size()];
    for (int i=0;i<images.size();i++){
         binfeatures[i] = new struct feature_node[randf.max_numtrees_*randf.num_landmark_+1];
    }
    
//    int bincode;
//    int ind;
//    int leafnode_per_tree = pow(2,(randf.max_depth_-1));
    
    Mat_<double> rotation;
    double scale;

    // extract feature for each samples
   // #pragma omp parallel for
    for (int i=0;i < images.size();i++){
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape_,rotation,scale);
       	#pragma omp parallel for
        for (int j =0; j <randf.num_landmark_; j++){
	       GetCodefromRandomForest(binfeatures[i], j*randf.max_numtrees_,randf.rfs_[j], images[i], current_shapes[i],
                                    bounding_boxs[i], rotation, scale);
//            for(int k = 0; k< randf.max_numtrees_;k++){
//                bincode = GetCodefromTree(randf.rfs_[j][k],images[i],current_shapes[i],bounding_boxs[i],rotation,scale);
//                ind = j * randf.max_numtrees_ + k;
//                binfeatures[i][ind].index = leafnode_per_tree * ind + bincode;
//                binfeatures[i][ind].value = 1;
//            }
            
        }
        binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].index = -1;
        binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].value = -1;
    }
    return binfeatures;
}
// get code of one landmark.
// index: the start index of tree.
void LBFRegressor::GetCodefromRandomForest(struct feature_node *binfeature,
                                           const int index,
                                           const vector<Tree>& rand_forest,
                                           const Mat_<uchar>& image,
                                           const Mat_<double>& shape,
                                           const BoundingBox& bounding_box,
                                           const Mat_<double>& rotation,
                                           const double scale){
    
    int leafnode_per_tree = pow(2,rand_forest[0].max_depth_-1);
    int landmark_x = shape(rand_forest[0].landmarkID_,0);
    int landmark_y = shape(rand_forest[0].landmarkID_,1);

    for (int iter = 0;iter<rand_forest.size();iter++){
        int currnode = 0;
        int bincode = 1;
        for(int i = 0;i<rand_forest[iter].max_depth_-1;i++){
            double x1 = rand_forest[iter].nodes_[currnode].feat[0];
            double y1 = rand_forest[iter].nodes_[currnode].feat[1];
            double x2 = rand_forest[iter].nodes_[currnode].feat[2];
            double y2 = rand_forest[iter].nodes_[currnode].feat[3];
            
            double project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
            double project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
            project_x1 = scale * project_x1 * bounding_box.width / 2.0;
            project_y1 = scale * project_y1 * bounding_box.height / 2.0;
            int real_x1 = (int)project_x1 + landmark_x;
            int real_y1 = (int)project_y1 + landmark_y;
            real_x1 = max(0,min(real_x1,image.cols-1));
            real_y1 = max(0,min(real_y1,image.rows-1));
            
            double project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
            double project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
            project_x2 = scale * project_x2 * bounding_box.width / 2.0;
            project_y2 = scale * project_y2 * bounding_box.height / 2.0;
            int real_x2 = (int)project_x2 + landmark_x;
            int real_y2 = (int)project_y2 + landmark_y;
            real_x2 = max(0,min(real_x2,image.cols-1));
            real_y2 = max(0,min(real_y2,image.rows-1));
            
            int pdf = (int)(image(real_y1,real_x1))-(int)(image(real_y2,real_x2));
            if (pdf < rand_forest[iter].nodes_[currnode].thresh){
                currnode =rand_forest[iter].nodes_[currnode].cnodes[0];
            }
            else{
                currnode =rand_forest[iter].nodes_[currnode].cnodes[1];
                bincode += pow(2, rand_forest[iter].max_depth_-2-i);
            }
        }
        binfeature[index+iter].index = leafnode_per_tree*(index+iter)+bincode;
        binfeature[index+iter].value = 1;
        
    }
}

void LBFRegressor::GlobalPrediction(struct feature_node** binfeatures,
                                    vector<Mat_<double> >& current_shapes,
                                    const vector<BoundingBox> & bounding_boxs,
                                    int stage){
    int num_train_sample = (int)current_shapes.size();
    int num_residual = current_shapes[0].rows*2;
    double tmp;
    double scale;
    Mat_<double>rotation;
    Mat_<double> deltashape_bar(num_residual/2,2);
   // #pragma omp parallel for
    for (int i=0;i<num_train_sample;i++){
        current_shapes[i] = ProjectShape(current_shapes[i],bounding_boxs[i]);
        //double t =(double)cvGetTickCount();
       	//#pragma omp parallel for
        for (int j=0;j<num_residual;j++){
            tmp = predict(Models_[stage][j],binfeatures[i]);
            if (j < num_residual/2){
                deltashape_bar(j,0) = tmp;
            }
            else{
                deltashape_bar(j-num_residual/2,1) = tmp;
            }
        }
        // transfer or not to be decided
        // now transfer
        SimilarityTransform(current_shapes[i],mean_shape_,rotation,scale);
        transpose(rotation,rotation);
        deltashape_bar = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((current_shapes[i]+deltashape_bar),bounding_boxs[i]);
    }
}

void LBFRegressor::ReleaseFeatureSpace(struct feature_node ** binfeatures,
                         int num_train_sample){
    for (int i = 0;i < num_train_sample;i++){
            delete[] binfeatures[i];
    }
    delete[] binfeatures;
}

Mat_<double>  LBFRegressor::Predict(const cv::Mat_<uchar>& image,
                                    const BoundingBox& bounding_box,
                                    int initial_num){
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > current_shapes;
    vector<BoundingBox>  bounding_boxs;


    images.push_back(image);
    bounding_boxs.push_back(bounding_box);
    current_shapes.push_back(ReProjectShape(mean_shape_, bounding_box));
   
//    Mat img = imread("/Users/lequan/workspace/LBF/Datasets/lfpw/testset/image_0078.png");
//    // draw result :: red
//    for(int j = 0;j < global_params_.landmark_num;j++){
//        circle(img,Point2d(current_shapes[0](j,0),current_shapes[0](j,1)),1,Scalar(255,255,255),-1,8,0);
//    }
//    imshow("result", img);
//    waitKey(0);
//    string name = "example mean.jpg";
//    imwrite(name,img);
    
    
    for ( int stage = 0; stage < global_params_.max_numstage; stage++){
        struct feature_node ** binfeatures ;
        binfeatures = DeriveBinaryFeat(RandomForest_[stage],images,current_shapes,bounding_boxs);
        GlobalPrediction(binfeatures, current_shapes,bounding_boxs,stage);
        ReleaseFeatureSpace(binfeatures, images.size());
        
//        Mat image = imread("/Users/lequan/workspace/LBF/Datasets/afw/image_0078.png");
//        // draw result :: red
//        for(int j = 0;j < global_params_.landmark_num;j++){
//            circle(image,Point2d(current_shapes[0](j,0),current_shapes[0](j,1)),1,Scalar(255,255,255),-1,8,0);
//        }
//        imshow("result", image);
//        waitKey(0);
//        string name = "example "+ to_string(stage) + ".jpg";
//        imwrite(name,image);

    }
    return current_shapes[0];
}

void LBFRegressor::Save(string modelPath, string regressorPath) {
    global_params_.modelPath = modelPath;
    global_params_.regressorPath = regressorPath;
    cout << endl<<"Saving model..." << endl;
    ofstream fout;
    fout.open(modelPath);
    // write the Regressor to file
    WriteGlobalParam(fout);
    WriteRegressor(fout);
    fout.close();
    cout << "End" << endl;

    
}

void LBFRegressor::Load(string modelPath, string regressorPath){
    cout << "Loading model from "<< modelPath  << endl;
    ifstream fin;
    fin.open(modelPath);
    ReadGlobalParam(fin);

    // set param of Regressor
    global_params_.modelPath = modelPath;
    global_params_.regressorPath = regressorPath;
    max_numstage_ = global_params_.max_numstage;
    RandomForest_.resize(max_numstage_);
    Models_.resize(max_numstage_);

    ReadRegressor(fin);
    fin.close();
    cout << "End"<<endl;
}
void  LBFRegressor::WriteGlobalParam(ofstream& fout){
    fout << global_params_.bagging_overlap << endl;
    fout << global_params_.max_numtrees << endl;
    fout << global_params_.max_depth << endl;
    fout << global_params_.max_numthreshs << endl;
    fout << global_params_.landmark_num << endl;
    fout << global_params_.initial_num << endl;
    fout << global_params_.max_numstage << endl;
    
    for (int i = 0; i< global_params_.max_numstage; i++){
        fout << global_params_.max_radio_radius[i] << " ";
        
    }
    fout << endl;
    
    for (int i = 0; i < global_params_.max_numstage; i++){
        fout << global_params_.max_numfeats[i] << " ";
    }
    fout << endl;
}

void  LBFRegressor::ReadGlobalParam(ifstream& fin){
    fin >> global_params_.bagging_overlap;
    fin >> global_params_.max_numtrees;
    fin >> global_params_.max_depth;
    fin >> global_params_.max_numthreshs;
    fin >> global_params_.landmark_num;
    fin >> global_params_.initial_num;
    fin >> global_params_.max_numstage;
    
    for (int i = 0; i< global_params_.max_numstage; i++){
        fin >> global_params_.max_radio_radius[i];
    }
    
    for (int i = 0; i < global_params_.max_numstage; i++){
        fin >> global_params_.max_numfeats[i];
    }
}
void  LBFRegressor::WriteRegressor(ofstream& fout){
    for(int i = 0;i < global_params_.landmark_num;i++){
        fout << mean_shape_(i,0)<<" "<< mean_shape_(i,1)<<" ";
    }
    fout<<endl;
    ofstream fout_reg;
    //fout_reg.open(modelPath + "/Regressor.model",ios::binary);
    fout_reg.open(global_params_.regressorPath, ios::binary);
    for (int i=0; i < global_params_.max_numstage; i++ ){
        RandomForest_[i].Write(fout);
        fout << Models_[i].size()<< endl;
        for (int j=0; j<Models_[i].size();j++){
            save_model_bin(fout_reg, Models_[i][j]);
        }
    }
    fout_reg.close();
}

void LBFRegressor::ReadRegressor(ifstream& fin){
    mean_shape_ = Mat::zeros(global_params_.landmark_num,2,CV_64FC1);
    for(int i = 0;i < global_params_.landmark_num;i++){
        fin >> mean_shape_(i,0) >> mean_shape_(i,1);
    }
    ifstream fin_reg;
    //fin_reg.open(modelPath + "/Regressor.model",ios::binary);
    fin_reg.open(global_params_.regressorPath, ios::binary);
    for (int i=0; i < global_params_.max_numstage; i++ ){
        RandomForest_[i].Init(global_params_);
        RandomForest_[i].Read(fin);
        int num =0;
        fin >> num;
        Models_[i].resize(num);
        for (int j=0;j<num;j++){
            Models_[i][j]   = load_model_bin(fin_reg);
        }
    }
    fin_reg.close();
}

