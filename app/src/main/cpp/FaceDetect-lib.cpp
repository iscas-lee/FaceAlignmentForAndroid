//
// Created by Administrator on 2017/12/15.
//

#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "FaceAlignmentForAndroid/FaceTracker.h"
#include <android/log.h>

#define LOG_TAG "jni.out"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

FaceTracker Face_Tracker;

#define FACEDETECT_METHOD(METHOD_NAME) \
  Java_zeng_com_opencv_1test03_FaceDetect_##METHOD_NAME  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeInit)(
    JNIEnv* env, jobject, jstring modelPath);


JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeFaceDetect)( JNIEnv* env, jobject,
    jintArray srcImg, jint width, jint height);

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeGetFaceNum)( JNIEnv* env, jobject);

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeGetFaceLandmark)( JNIEnv* env, jobject,
    jint index, jintArray res);

#ifdef __cplusplus
}
#endif

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeInit)(JNIEnv* env, jobject, 
    jstring modelPath) {
    LOGI("face_tracker init");
    const char* mp = env->GetStringUTFChars(modelPath, JNI_FALSE);
    std::string model_path = mp;
    Face_Tracker.init(model_path,1,DETECT_DLIB,SHAPE_DLIB);
    model_path = "model_path: " + model_path;
    LOGI("%s",model_path.data());
    return 1;
}

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeFaceDetect)( JNIEnv* env, jobject,
    jintArray srcImg, jint width, jint height) {
    //LOGI("face_tracker start detect");

    jint *srcbuf = env->GetIntArrayElements(srcImg, JNI_FALSE);
    cv::Mat srcImgMat(height, width, CV_8UC4, (unsigned char *) srcbuf);

    Face_Tracker.faceAlignment(srcImgMat,IMG_BGRA);

    //jintArray face_landmark = env->NewIntArray(68*2);
    //jint* fl_ptr = env->GetIntArrayElements(res_faceLandmark, JNI_FALSE);
    //LOGI("face_tracker end detect");
    return Face_Tracker.faces_shapes_.size();


//    if(Face_Tracker.faces_shapes_.size() == 0) return 0;
//    for(int i=0;i<68;i++) {
//        *(fl_ptr + i*2)     = (jint)Face_Tracker.faces_shapes_[0](i).row;
//        *(fl_ptr + i*2 + 1) = (jint)Face_Tracker.faces_shapes_[0](i).col;
//    }
}

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeGetFaceNum)( JNIEnv* env, jobject) {
    return Face_Tracker.faces_shapes_.size();
}

JNIEXPORT jint JNICALL
FACEDETECT_METHOD(nativeGetFaceLandmark)( JNIEnv* env, jobject,
    jint Jindex, jintArray res_faceLandmark) {
    int index = Jindex;
    jint* res_ptr = env->GetIntArrayElements(res_faceLandmark, JNI_FALSE);

    if(index>Face_Tracker.faces_shapes_.size()) return 0;

    jint res_size = env->GetArrayLength(res_faceLandmark);
    if((int)res_size!=Face_Tracker.faces_shapes_[index].landmark_num()*2) return -1;

    for(int i=0;i<res_size/2;i++) {
        *(res_ptr + i*2)     = (jint)Face_Tracker.faces_shapes_[index](i).row;
        *(res_ptr + i*2 + 1) = (jint)Face_Tracker.faces_shapes_[index](i).col;
    }
    return 1;

}
