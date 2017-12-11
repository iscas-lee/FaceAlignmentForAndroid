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
#include "FaceShape.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>


enum IMG_CODE {
    IMG_GRAY = 0,
    IMG_RGB = 1,
    IMG_BGRA = 2,
};

enum SHAPE_METHOD {
    SHAPE_LBF3000 = 0,
    SHAPE_DLIB = 1,
};

enum DETECT_METHOD {
    DETECT_OPENCV = 0,
    DETECT_DLIB = 1,
};

class FaceTracker {
private:
    LBFRegressor lbf_regressor_;
    cv::CascadeClassifier opencv_cascade_;

    dlib::frontal_face_detector dlib_detector_;
    dlib::shape_predictor dlib_shape_pred_;
    
    DETECT_METHOD detect_method_;
    SHAPE_METHOD shape_method_;

    int smallImg_row_;
    int faces_max_num_;  
    //std::vector<std::vector<KalmanParam[2]> > kalman_params_;

    int cvImg2Code(IMG_CODE srcCode, IMG_CODE resCode);

public:
    std::vector<BoundingBox> faces_boxes_;
    std::vector<FaceShape> faces_shapes_;
    bool Kalman_state_;

    FaceTracker() {
        smallImg_row_ = 300;
        faces_max_num_ = 5;
        Kalman_state_ = false; 
    }
    ~FaceTracker() {

    }
    
    // 初始化
    void init(  std::string modelPath, 
                int faces_max_num = 5, 
                DETECT_METHOD detect_method = DETECT_OPENCV, 
                SHAPE_METHOD shape_method = SHAPE_LBF3000);

    
    // 人脸检测
    std::vector<BoundingBox> calculateFaceBox(cv::Mat& srcImg, IMG_CODE srcCode);
    // 人脸特征点识别
    FaceShape calculateFaceShape(cv::Mat& srcImg, BoundingBox face_box, IMG_CODE srcCode);

    void faceAlignAndDraw(cv::Mat& srcImg, cv::Mat& resImg, IMG_CODE srcCode, IMG_CODE resCode);
    void faceAlignment(cv::Mat& srcImg, IMG_CODE srcCode);

    std::vector<BoundingBox> getAllFaceBox();
    BoundingBox getOneFaceBox();

    void updateImage(cv::Mat grayImg);
    std::vector<cv::Mat_<double> > getAllFaceShape();
    cv::Mat_<double> getOneFaceShape();

//    void updateShapeByKalman(cv::Mat_<double>& face_shape, int index);


    void colorConvert(cv::Mat& srcImg, cv::Mat& resImg, IMG_CODE srcCode, IMG_CODE resCode);


};


#endif