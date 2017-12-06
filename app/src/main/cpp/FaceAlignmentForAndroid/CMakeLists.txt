
AUX_SOURCE_DIRECTORY(. FaceAlignment_SRC)
AUX_SOURCE_DIRECTORY(./liblinear liblinear_SRC)
AUX_SOURCE_DIRECTORY(./liblinear/blas liblinearBlas_SRC)

set(OpenCV_DIR ../../../../../OpenCV-android-sdk/sdk/native/jni)

find_package(OpenCV REQUIRED)

ADD_LIBRARY(FaceAlignment ${FaceAlignment_SRC} ${liblinear_SRC} ${liblinearBlas_SRC})

target_link_libraries(${OpenCV_LIBS})