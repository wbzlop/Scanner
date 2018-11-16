package com.document.pdf.scanner.filter;

import android.graphics.Bitmap;

import jp.co.cyberagent.android.gpuimage.GPUImageAlphaBlendFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageColorInvertFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageContrastFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageFilterGroup;
import jp.co.cyberagent.android.gpuimage.GPUImageGaussianBlurFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageHardLightBlendFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageSaturationFilter;
import jp.co.cyberagent.android.gpuimage.GPUImageWeakPixelInclusionFilter;

/**
 * Created by wubaozeng on 2018/9/28.
 * 黑白滤镜
 */

public class GPUImageBlackWhiteFilter extends GPUImageFilterGroup {

    private GPUImageHardLightBlendFilter blendFilter;

    public GPUImageBlackWhiteFilter() {
        super();

        //防止坐标翻转
        this.mFilters.add(new GPUImageFilter());
        //边缘识别
        this.mFilters.add(new GPUImageLaplacianForTextFilter(2f));
        //像素补偿
        this.mFilters.add(new GPUImageWeakPixelInclusionFilter());
        //反色
        this.mFilters.add(new GPUImageColorInvertFilter());
        //高斯模糊
        this.mFilters.add(new GPUImageGaussianBlurFilter(0.2f));

        //强光混合
        blendFilter = new GPUImageHardLightBlendFilter();
        this.mFilters.add(blendFilter);
        //饱和度
        this.mFilters.add(new GPUImageSaturationFilter(0));
        //对比度
//        this.mFilters.add(new GPUImageContrastFilter(1.15f));
        updateMergedFilters();

    }

    /**
     * 设置混合图片
     *
     * @param bitmap
     */
    public void setBlendBitmap(Bitmap bitmap) {
        blendFilter.onDestroy();
        blendFilter.setBitmap(bitmap);
    }

}
