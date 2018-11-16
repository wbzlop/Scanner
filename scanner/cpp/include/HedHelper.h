//
// Created by wbz on 2018/11/8.
//

#ifndef SCANNER2_HEDHELPER_H
#define SCANNER2_HEDHELPER_H

#include <jni.h>
#include <opencv2/opencv.hpp>
using namespace cv;


void bitmap_to_floatArray(JNIEnv *env, jobject &srcBitmap,jfloatArray &floatArray);
void floatArray_to_bitmap(JNIEnv *env,jfloatArray &floatArray,jobject &dstBitmap);


//https://raw.githubusercontent.com/opencv/opencv_attic/master/opencv/modules/java/generator/src/cpp/Mat.cpp
template<typename T> int mat_get(cv::Mat* m, int row, int col, int count, char* buff);
template<typename T> int mat_put(cv::Mat* m, int row, int col, int count, char* buff);

#endif //SCANNER2_HEDHELPER_H
