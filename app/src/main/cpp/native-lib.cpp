#include <jni.h>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "FaceAlignmentForAndroid/FaceTracker.h"

FaceTracker face_tracker;

extern "C"
JNIEXPORT jstring
JNICALL
Java_zeng_com_opencv_1test03_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello="111" ;
    hello = "222";
    std::ifstream infile;
    infile.open("/storage/emulated/0/1.txt");
    if(infile.is_open()){
        getline(infile, hello);
        hello = hello+" read 1.txt";
    }
    else
        hello = "can't read 1.txt";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT jint
JNICALL
Java_zeng_com_opencv_1test03_MainActivity_rgb2Gray(
        JNIEnv *env, jobject , jintArray srcImg, jintArray resImg, int w, int h) {
    jint *srcbuf = env->GetIntArrayElements(srcImg, JNI_FALSE);
    cv::Mat srcImgMat(h, w, CV_8UC4, (unsigned char *) srcbuf);

    jint *resbuf = env->GetIntArrayElements(resImg, JNI_FALSE);
    cv::Mat resImgMat(h, w, CV_8UC4, (unsigned char *) resbuf);

    cv::Mat imgGray(h, w,  CV_8UC1);

    cv::Mat rgbImgMat = cv::imread("/storage/emulated/0/people.jpg");

    face_tracker.ColorConvert(srcImgMat,imgGray,IMG_BGRA,IMG_GRAY);
    //face_tracker.ColorConvert(rgbImgMat,imgGray,IMG_RGB,IMG_GRAY);
    face_tracker.ColorConvert(imgGray,resImgMat,IMG_GRAY,IMG_BGRA);



    return 1;
}
