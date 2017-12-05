

#include "FaceTracker.h"

using namespace std;

void FaceTracker::Init(std::string modelPath, TRACK_METHOD method = FACE_3000FPS, int faces_max_num = 5) {
    method_ = method;
    faces_max_num_ = faces_max_num;
    lbf_regressor_.Load(modelPath+"LBF.model", modelPath+"Regressor.model");
    face_cascade_.load(modelPath+"haarcascade_frontalface_alt.xml");
}

vector<BoundingBox> FaceTracker::FaceDetect(cv::Mat grayImg) { 
    vector<cv::Rect> faces;
    vector<BoundingBox> faces_boxes;

    float scale = smallImg_row_*1.0/grayImg.rows;
    cv::Mat smallImg( smallImg_row_, cvRound(grayImg.cols * scale), CV_8UC1 );
    resize( grayImg, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    face_cascade_.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        cv::Size(30, 30) );
    for( vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++){
        cv::Point center;
        //cv::Scalar color = colors[i%8];
        BoundingBox boundingbox;
        
        boundingbox.start_x = r->x*scale;
        boundingbox.start_y = r->y*scale;
        boundingbox.width   = (r->width-1)*scale;
        boundingbox.height  = (r->height-1)*scale;
        boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
        boundingbox.centroid_y = boundingbox.start_y + boundingbox.height/2.0;

        faces_boxes.push_back(boundingbox);
    }

    return faces_boxes;
}


cv::Mat_<double> FaceTracker::FaceShape(cv::Mat grayImg, BoundingBox face_box) { 
    cv::Mat_<double> current_shape = lbf_regressor_.Predict(grayImg,face_box,1);
    return current_shape;
}

void FaceTracker::ColorConvert(cv::Mat srcImg, cv::Mat resImg, IMG_CODE srcCode, IMG_CODE resCode) {
    int cv2code = cvImg2Code(srcCode, resCode);
    if(cv2code==-1)
        srcImg.copyTo(resImg);
    else
        cv::cvtColor(srcImg, resImg, cv2code);
    
}

int FaceTracker::cvImg2Code(IMG_CODE srcCode, IMG_CODE resCode) {
    int code2code[3][3] = { {-1,            CV_GRAY2RGB,    CV_GRAY2BGRA},
                            {CV_RGB2GRAY,   -1,             CV_RGB2BGRA},
                            {CV_BGRA2GRAY,  CV_BGRA2RGB,    -1}};
    return code2code[srcCode][resCode];
}


void FaceTracker::UpdateShapeByKalman(cv::Mat_<double> face_shape, int index) { 
    int landmark_num = face_shape.rows;
    //double p1;
    //std::vector<KalmanParam[2]> *kalman_ptr = &kalman_params_[index]; 
    for(int i=0;i<landmark_num; i++) {
        for(int j=0; j<2; j++) {
            float pre_optimal = faces_shapes_[index](i,j);
            faces_shapes_[index](i,j) = kalman_params_[index][i][j].KalmanUpdate(face_shape(i,j), pre_optimal);
        }
    }
}