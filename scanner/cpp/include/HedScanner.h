//
// Created by wbz on 2018/11/8.
//

#ifndef SCANNER2_HEDSCANNER_H
#define SCANNER2_HEDSCANNER_H

#include <jni.h>
#include <opencv2/opencv.hpp>

namespace hedScanner {
    /**
 RefLineVec4i 比较特殊，如果把它看成向量的话，它的反向是遵守一定的规则的。
 RefLineVec4i 里面的两个点，总是从左往右的方向，如果 RefLine 和 Y 轴平行(区分不了左右)，则按照从下往上的方向
 */
    typedef cv::Vec4i RefLineVec4i;
    struct Corner {
        cv::Point point;
        std::vector<cv::Vec4i> segments;
    };



    class HedScanner {


    public:


        HedScanner();

        virtual ~HedScanner();

        static std::vector<cv::Point> scanPoint(JNIEnv *env, jfloatArray hedData);


    private:


    };

}

#endif //SCANNER2_HEDSCANNER_H
