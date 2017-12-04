//
//  FaceTracker.h
//  my LBFRegressor for android
//
//  Created by zeng on 12/01/17.
//  Copyright (c) 2015 zeng. All rights reserved.
//

#ifndef __FaceTracker_H__
#define __FaceTracker_H__

#include "LBFRegressor.h"

enum IMG_CODE {
    IMG_GRAY = 0,
    IMG_RGB = 1,
    IMG_BGRA = 2,
};

enum TRACK_METHOD {
    FACE_3000FPS = 0,
    FACE_DLIB = 1,
};

class KalmanParam {
public:
    static float R;
    static float Q;
    float K;
    float P;
    float x;
    KalmanParam() {
        R=0.01;
        Q=0.0001;
    }
};

class FaceTracker {
private:
    LBFRegressor lbf_regressor_;
    cv::CascadeClassifier face_cascade_;
    TRACK_METHOD method_;

    int smallImg_row_;
    int faces_max_num_;

    std::vector<BoundingBox> faces_boxes_;
    std::vector<cv::Mat_<double> > faces_shapes_;
    std::vector<std::vector<KalmanParam[2]> > kalman_params_;

    int cvImg2Code(IMG_CODE srcCode, IMG_CODE resCode);


public:
    FaceTracker() {
        smallImg_row_ = 500;
        faces_max_num_ = 5;

    }
    ~FaceTracker() {

    }
    
    // 初始化
    void Init(std::string modelPath, TRACK_METHOD method = FACE_3000FPS, int faces_max_num = 5);
    //void Init(std::string modelPath);

    // 人脸检测
    std::vector<BoundingBox> FaceDetect(cv::Mat grayImg);
    // 人脸特征点识别
    cv::Mat_<double> FaceShape(cv::Mat grayImg, BoundingBox face_box);

    void UpdataImage(cv::Mat grayImg);
    std::vector<cv::Mat_<double> > GetAllFaceShape();
    cv::Mat_<double> GetOneFaceShape();

    void KalmanFilter(cv::Mat_<double>& face_shape, std::vector<KalmanParam[2]>& kalman_param);


    void ColorConvert(cv::Mat srcImg, cv::Mat resImg, IMG_CODE srcCode, IMG_CODE resCode);


};


#endif