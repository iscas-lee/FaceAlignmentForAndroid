#include <jni.h>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "FaceAlignmentForAndroid/FaceTracker.h"

FaceTracker face_tracker;
int filename_i = 0;
std::string modelpath = "/storage/emulated/0/faceTest/";

extern "C"
JNIEXPORT jint
JNICALL
Java_zeng_com_opencv_1test03_MainActivity_faceTrackerInit(
        JNIEnv *env,
        jobject /* this */) {
    face_tracker.init(modelpath+"Model/");
    return 0;
}

extern "C"
JNIEXPORT jstring
JNICALL
Java_zeng_com_opencv_1test03_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */,int num) {
    std::string filename;
    std::ostringstream s1;
    s1 << 1000+num;
    filename = s1.str().substr(1,3)+".jpg";
    return env->NewStringUTF(filename.c_str());
}

extern "C"
JNIEXPORT jint
JNICALL
Java_zeng_com_opencv_1test03_MainActivity_rgb2Gray(
        JNIEnv *env, jobject , jintArray srcImg, jintArray resImg, int w, int h, int num) {
    jint *srcbuf = env->GetIntArrayElements(srcImg, JNI_FALSE);
    cv::Mat srcImgMat(h, w, CV_8UC4, (unsigned char *) srcbuf);

    jint *resbuf = env->GetIntArrayElements(resImg, JNI_FALSE);
    cv::Mat resImgMat(h, w, CV_8UC4, (unsigned char *) resbuf);

    cv::Mat imgGray(h, w,  CV_8UC1);

    std::string filename;
    std::ostringstream s1;
    s1 << 1000+num;
    filename = s1.str().substr(1,3)+".jpg";

    cv::Mat rgbImgMatTemp = cv::imread(modelpath+"test/"+filename);
    cv::Mat rgbImgMat(h, w, CV_8UC3);
    cv::resize(rgbImgMatTemp,rgbImgMat,rgbImgMat.size(),0,0,cv::INTER_LINEAR);

    //face_tracker.ColorConvert(srcImgMat,imgGray,IMG_BGRA,IMG_GRAY);
    //face_tracker.ColorConvert(rgbImgMat,imgGray,IMG_RGB,IMG_GRAY);
    //face_tracker.ColorConvert(rgbImgMat,resImgMat,IMG_RGB,IMG_BGRA);
    face_tracker.faceAlignAndDraw(rgbImgMat,resImgMat,IMG_RGB,IMG_BGRA);



    return 1;
}
