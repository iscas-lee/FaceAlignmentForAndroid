
//
//  FaceTracker.h
//  my LBFRegressor for android
//
//  Created by zeng on 12/01/17.
//  Copyright (c) 2015 zeng. All rights reserved.
//

#ifndef __FaceShape_H__
#define __FaceShape_H__

#define FACE_landmark_NUM 68

#include <vector>
#include <opencv2/opencv.hpp>
// #include <dlib/opencv.h>
// #include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/image_processing/render_face_detections.h>
// #include <dlib/image_processing.h>

class KalmanParam {
public:
    float R;
    float Q;
    float K;
    float P;
    float optX;
    KalmanParam() {
        R=0.1;
        Q=0.01;
        P = 10;
        K = 1;
    }

    float calculateOpt(float observe, float pre_opt) {
        float opt = pre_opt + K*(observe - pre_opt);
        return opt;
    }

    void updateParam() {
        float p1 = P + Q;
        K = p1/(p1+R);
        //float opt = pre_opt + K*(observe - pre_opt);
        P = (1-K)*p1;
    }
};

class PixelCoordinate {
public:
    int row;
    int col;
    //KalmanParam kalman_param[2];

    PixelCoordinate() {
        row=0;
        col=0;
    }

    void set(int r, int c) {
        row = r; col = c;
    }

};

class FaceShape {
public:
    std::vector<PixelCoordinate> pixel_vector;
    KalmanParam kalman_param;
    int kalman_update_times;

    FaceShape() {
        //pixel_vector.resize(FACE_landmark_NUM);
        kalman_update_times = 0;
    }

    int landmark_num() {
        return pixel_vector.size();
    }

    void set(cv::Mat_<double>& shape_mat) { 
        pixel_vector.resize(shape_mat.rows);
        for(int i=0; i<shape_mat.rows; i++) {
            pixel_vector[i].row = shape_mat(i,0);
            pixel_vector[i].col = shape_mat(i,1);
        }
        kalman_update_times = 0;
    }

    // void set(dlib::full_object_detection& shape_dlib) {
    //     pixel_vector.resize(shape_dlib.num_parts());
    //     for(int i=0; i<shape_dlib.num_parts(); i++) {
    //         pixel_vector[i].row = shape_dlib.part(i).x();
    //         pixel_vector[i].col = shape_dlib.part(i).y();
    //     }
    // }

    bool updateByKalman(cv::Mat_<double>& shape_mat) {
        if(shape_mat.rows == pixel_vector.size()) {
            kalman_param.updateParam();            
            for(int i=0; i<shape_mat.rows; i++) {
                PixelCoordinate pre_opt = pixel_vector[i];
                pixel_vector[i].row = kalman_param.calculateOpt(shape_mat(i,0), pre_opt.row);
                pixel_vector[i].col = kalman_param.calculateOpt(shape_mat(i,0), pre_opt.col);
            }
            kalman_update_times++;
            return true;
        }
        else  return false;
    }

    PixelCoordinate& operator() (int index) {
        return pixel_vector[index];
    }
};

#endif