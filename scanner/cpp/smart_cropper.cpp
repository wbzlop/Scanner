//
// Created by qiulinmin on 8/1/17.
//
#include <jni.h>
#include <string>
#include <android_utils.h>
#include <Scanner.h>
#include <HedHelper.h>
#include "HedScanner.h"
#include "android/log.h"

static const char* const kClassDocScanner = "me/pqpo/smartcropperlib/SmartCropper";

static struct {
    jclass jClassPoint;
    jmethodID jMethodInit;
    jfieldID jFieldIDX;
    jfieldID jFieldIDY;
} gPointInfo;

static void initClassInfo(JNIEnv *env) {
    gPointInfo.jClassPoint = reinterpret_cast<jclass>(env -> NewGlobalRef(env -> FindClass("android/graphics/Point")));
    gPointInfo.jMethodInit = env -> GetMethodID(gPointInfo.jClassPoint, "<init>", "(II)V");
    gPointInfo.jFieldIDX = env -> GetFieldID(gPointInfo.jClassPoint, "x", "I");
    gPointInfo.jFieldIDY = env -> GetFieldID(gPointInfo.jClassPoint, "y", "I");
}

static jobject createJavaPoint(JNIEnv *env, Point point_) {
    return env -> NewObject(gPointInfo.jClassPoint, gPointInfo.jMethodInit, point_.x, point_.y);
}

static void native_scan(JNIEnv *env, jclass type, jobject srcBitmap, jobjectArray outPoint_) {
    if (env -> GetArrayLength(outPoint_) != 4) {
        return;
    }
    Mat srcBitmapMat;
    bitmap_to_mat(env, srcBitmap, srcBitmapMat);
    Mat bgrData(srcBitmapMat.rows, srcBitmapMat.cols, CV_8UC3);
    cvtColor(srcBitmapMat, bgrData, CV_RGBA2BGR);
    scanner::Scanner docScanner(bgrData);
    std::vector<Point> scanPoints = docScanner.scanPoint();
    if (scanPoints.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scanPoints[i]));
        }
    }
}

static vector<Point> pointsToNative(JNIEnv *env, jobjectArray points_) {
    int arrayLength = env->GetArrayLength(points_);
    vector<Point> result;
    for(int i = 0; i < arrayLength; i++) {
        jobject point_ = env -> GetObjectArrayElement(points_, i);
        int pX = env -> GetIntField(point_, gPointInfo.jFieldIDX);
        int pY = env -> GetIntField(point_, gPointInfo.jFieldIDY);
        result.push_back(Point(pX, pY));
    }
    return result;
}

static void native_crop(JNIEnv *env, jclass type, jobject srcBitmap, jobjectArray points_, jobject outBitmap) {
    std::vector<Point> points = pointsToNative(env, points_);
    if (points.size() != 4) {
        return;
    }
    Point leftTop = points[0];
    Point rightTop = points[1];
    Point rightBottom = points[2];
    Point leftBottom = points[3];

    Mat srcBitmapMat;
    bitmap_to_mat(env, srcBitmap, srcBitmapMat);

    AndroidBitmapInfo outBitmapInfo;
    AndroidBitmap_getInfo(env, outBitmap, &outBitmapInfo);
    Mat dstBitmapMat;
    int newHeight = outBitmapInfo.height;
    int newWidth = outBitmapInfo.width;
    dstBitmapMat = Mat::zeros(newHeight, newWidth, srcBitmapMat.type());

    vector<Point2f> srcTriangle;
    vector<Point2f> dstTriangle;

    srcTriangle.push_back(Point2f(leftTop.x, leftTop.y));
    srcTriangle.push_back(Point2f(rightTop.x, rightTop.y));
    srcTriangle.push_back(Point2f(leftBottom.x, leftBottom.y));
    srcTriangle.push_back(Point2f(rightBottom.x, rightBottom.y));

    dstTriangle.push_back(Point2f(0, 0));
    dstTriangle.push_back(Point2f(newWidth, 0));
    dstTriangle.push_back(Point2f(0, newHeight));
    dstTriangle.push_back(Point2f(newWidth, newHeight));

    Mat transform = getPerspectiveTransform(srcTriangle, dstTriangle);
    warpPerspective(srcBitmapMat, dstBitmapMat, transform, dstBitmapMat.size());

    mat_to_bitmap(env, dstBitmapMat, outBitmap);
}

static void native_bitmap_to_floats(JNIEnv *env, jclass type, jobject srcBitmap, jfloatArray floats)
{

    bitmap_to_floatArray(env,srcBitmap,floats);

}

static void native_floats_to_bitmap(JNIEnv *env, jclass type,  jfloatArray floats,jobject srcBitmap)
{

    floatArray_to_bitmap(env,floats,srcBitmap);

}

static void native_hed_scan(JNIEnv *env, jclass type,  jfloatArray floats, int originalWidth,int originalHeight,jobjectArray outPoint_)
{
    std::vector<Point> scanPoints = hedScanner::HedScanner::scanPoint(env,floats);

    if(scanPoints.size() == 4)
    {
        for(int i = 0; i < scanPoints.size(); i++) {
            cv::Point cv_point = scanPoints[i];

            cv::Point scaled_point = cv::Point(cv_point.x * originalWidth / 256, cv_point.y * originalHeight / 256);
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scaled_point));

        }
    } else
    {
        for(int i = 0; i < 4; i++) {

            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, cv::Point(0,0)));

        }
    }

}

static JNINativeMethod gMethods[] = {

        {
                "nativeScan",
                "(Landroid/graphics/Bitmap;[Landroid/graphics/Point;)V",
                (void*)native_scan
        },

        {
                "nativeCrop",
                "(Landroid/graphics/Bitmap;[Landroid/graphics/Point;Landroid/graphics/Bitmap;)V",
                (void*)native_crop
        }
        ,

        {
                "nativeBitmapToFloatArray",
                "(Landroid/graphics/Bitmap;[F)V",
                (void*)native_bitmap_to_floats
        }
        ,

        {
                "nativeFloatArrayToBitmap",
                "([FLandroid/graphics/Bitmap;)V",
                (void*)native_floats_to_bitmap
        }
        ,

        {
                "nativeHEDScan",
                "([FII[Landroid/graphics/Point;)V",
                (void*)native_hed_scan
        }

};

extern "C"
JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv *env = NULL;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        return JNI_FALSE;
    }
    jclass classDocScanner = env->FindClass(kClassDocScanner);
    if(env -> RegisterNatives(classDocScanner, gMethods, sizeof(gMethods)/ sizeof(gMethods[0])) < 0) {
        return JNI_FALSE;
    }
    initClassInfo(env);
    return JNI_VERSION_1_4;
}
