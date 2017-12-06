#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "FaceAlignmentForAndroid/FaceTracker.h"

FaceTracker face_tracker;

extern "C"
JNIEXPORT jstring
JNICALL
Java_zeng_com_opencv_1test03_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
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

    face_tracker.ColorConvert(srcImgMat,imgGray,IMG_BGRA,IMG_GRAY);
    face_tracker.ColorConvert(imgGray,resImgMat,IMG_GRAY,IMG_BGRA);



    return 1;
}
