

#include "FaceTracker.h"

using namespace std;


void FaceTracker::init(std::string modelPath, int faces_max_num, DETECT_METHOD detect_method, SHAPE_METHOD shape_method) {
    detect_method_ = detect_method;
    shape_method_ = shape_method;
    faces_max_num_ = faces_max_num;

    if(detect_method_ == DETECT_OPENCV) {
        opencv_cascade_.load(modelPath+"haarcascade_frontalface_alt.xml");
    }
    else if(detect_method_ == DETECT_DLIB) {
        dlib_detector = dlib::get_frontal_face_detector();
    }

    if(shape_method_ == SHAPE_LBF3000) {
        lbf_regressor_.Load(modelPath+"LBF.model", modelPath+"Regressor.model");
    }
    else if(shape_method_ == SHAPE_DLIB) {
        dlib::deserialize(modelPath +"shape_predictor_68_face_landmarks.dat") >> dlib_shape_pred;
    }   
}

vector<BoundingBox> FaceTracker::calculateFaceBox(cv::Mat& grayImg) { 
    vector<cv::Rect> faces;
    vector<BoundingBox> faces_boxes;

    float scale = grayImg.rows*1.0/smallImg_row_;
    cv::Mat smallImg( smallImg_row_, cvRound(grayImg.cols / scale), CV_8UC1 );
    resize( grayImg, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    opencv_cascade_.detectMultiScale( smallImg, faces,
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


FaceShape FaceTracker::calculateFaceShape(cv::Mat& grayImg, BoundingBox face_box) {
    cv::Mat_<double> shape_mat = lbf_regressor_.Predict(grayImg,face_box,1);
    FaceShape face_shape;
    face_shape.set(shape_mat);
    return face_shape;
}

void FaceTracker::colorConvert(cv::Mat& srcImg, cv::Mat& resImg, IMG_CODE srcCode, IMG_CODE resCode) {
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


void FaceTracker::faceAlignAndDraw(cv::Mat& srcImg, cv::Mat& resImg, IMG_CODE srcCode, IMG_CODE resCode) { 
    cv::Mat grayImg(srcImg.rows,srcImg.cols,CV_8UC1),rgbImg(srcImg.rows,srcImg.cols,CV_8UC3);
    colorConvert(srcImg,grayImg,srcCode,IMG_GRAY);
    colorConvert(srcImg,rgbImg,srcCode,IMG_RGB);

    vector<BoundingBox> faces_boxes = calculateFaceBox(grayImg);

    for(int i=0; i<faces_boxes.size(); i++) {
        BoundingBox boundingbox = faces_boxes[i];
        cv::rectangle(rgbImg, cvPoint(boundingbox.start_x,boundingbox.start_y),
                cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),cv::Scalar(0,255,0), 1, 8, 0);
        FaceShape current_shape = calculateFaceShape(grayImg,boundingbox);
        for(int i = 0;i < current_shape.landmark_num();i++){
            cv::circle(rgbImg, cv::Point2d(current_shape(i).row, current_shape(i).col), 3, cv::Scalar(255,255,255),-1,8,0);
        }
    }

    colorConvert(rgbImg, resImg, IMG_RGB, resCode);


}