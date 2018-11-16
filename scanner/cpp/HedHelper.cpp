//
// Created by wbz on 2018/11/8.
//

#include "HedHelper.h"
#include "android_utils.h"
#include "HedScanner.h"


void bitmap_to_floatArray(JNIEnv *env, jobject &srcBitmap, jfloatArray &floatArray)
{
    //1.bitmap_to_mat
    Mat original;
    bitmap_to_mat(env,srcBitmap,original);

    //2.resize
    Mat resizeMat;
    resize(original,resizeMat,Size(256,256),0,0);

    //3.去除 Alpha
    Mat rgbMat;
    cvtColor(resizeMat,rgbMat,COLOR_RGBA2BGR);

    //4.convert pixel type from int to float, and value range from (0, 255) to (0.0, 1.0)
    Mat floatMat;
    rgbMat.convertTo(floatMat,CV_32FC3,1.0 / 255);

    //5.get float value from mat
    char* values = (char*)env->GetPrimitiveArrayCritical(floatArray, 0);
    int res = mat_get<float>(&floatMat, 0, 0, 256*256*3, values);
    env->ReleasePrimitiveArrayCritical(floatArray, values, 0);

}


void floatArray_to_bitmap(JNIEnv *env,jfloatArray &floatArray,jobject &dstBitmap)
{
    Mat floatMat = Mat(256,256,CV_32FC1);
    char* values = (char*)env->GetPrimitiveArrayCritical(floatArray, 0);
    int res = mat_put<float>(&floatMat, 0, 0, 256*256*3, values);
    env->ReleasePrimitiveArrayCritical(floatArray, values, 0);

    Mat intMat;
    floatMat.convertTo(intMat,CV_8UC1,255.0);

    mat_to_bitmap(env,intMat,dstBitmap);
}

template<typename T> int mat_put(cv::Mat* m, int row, int col, int count, char* buff)
{
    if(! m) return 0;
    if(! buff) return 0;

    count *= sizeof(T);
    int rest = ((m->rows - row) * m->cols - col) * m->elemSize();
    if(count>rest) count = rest;
    int res = count;

    if( m->isContinuous() )
    {
        memcpy(m->ptr(row, col), buff, count);
    } else {
        // row by row
        int num = (m->cols - col) * m->elemSize(); // 1st partial row
        if(count<num) num = count;
        uchar* data = m->ptr(row++, col);
        while(count>0){
            memcpy(data, buff, num);
            count -= num;
            buff += num;
            num = m->cols * m->elemSize();
            if(count<num) num = count;
            data = m->ptr(row++, 0);
        }
    }
    return res;
}


template<typename T> int mat_get(cv::Mat* m, int row, int col, int count, char* buff)
{
    if(! m) return 0;
    if(! buff) return 0;

    int bytesToCopy = count * sizeof(T);
    int bytesRestInMat = ((m->rows - row) * m->cols - col) * m->elemSize();
    if(bytesToCopy > bytesRestInMat) bytesToCopy = bytesRestInMat;
    int res = bytesToCopy;

    if( m->isContinuous() )
    {
        memcpy(buff, m->ptr(row, col), bytesToCopy);
    } else {
        // row by row
        int bytesInRow = (m->cols - col) * m->elemSize(); // 1st partial row
        while(bytesToCopy > 0)
        {
            int len = std::min(bytesToCopy, bytesInRow);
            memcpy(buff, m->ptr(row, col), len);
            bytesToCopy -= len;
            buff += len;
            row++;
            col = 0;
            bytesInRow = m->cols * m->elemSize();
        }
    }
    return res;
}