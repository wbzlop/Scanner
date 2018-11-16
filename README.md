# Scanner
文档扫描功能中的滤镜及边框识别

## 滤镜
初期使用 ColorMatrix 实现，但存在以下问题：<br>
1.图片噪点多，ColorMatrix 无法去除噪点；<br>
2.CPU 处理，效率低下；<br>

### 解决方案
[GPUImage](https://github.com/cats-oss/android-gpuimage)<br>
它提供的 GPUImageFilterGroup 可让使用者自由组合滤镜效果。<br>
这里只提供最受欢迎的滤镜（统计结果）—— 黑白滤镜。笔者针对文字修改了滤波算子，该滤镜可以保留文字细节，同时过滤噪点和阴影。<br>
```java
private GLSurfaceCompatView surfaceView;
private GPUImage mGPUImage;

......

mGPUImage = new GPUImage(getActivity());
surfaceView = getActivity().findViewById(R.id.filter_view);
surfaceView.setWH(originalBitmap.getWidth(), originalBitmap.getHeight());
mGPUImage.setGLSurfaceView(surfaceView);
mGPUImage.setScaleType(GPUImage.ScaleType.CENTER_INSIDE);
mGPUImage.setImage(originalBitmap);

......

GPUImageBlackWhiteFilter filter = new GPUImageBlackWhiteFilter();
filter.setBlendBitmap(originalBitmap);
mGPUImage.setFilter(filter);
```


## 边框识别
初期使用了[SmartCropper](https://github.com/pqpo/SmartCropper)(博客地址这里：[Android 端基于 OpenCV 的边框识别功能](https://pqpo.me/2017/09/11/opencv-border-recognition/))<br>
边框识别存在以下问题：<br>
1.圆角矩形由于approxPolyDP 操作被破坏，尝试使用HoughLinesP操作去修正，但由于第二个问题，放弃；<br>
2.canny 非常受图片环境影响，稍微复杂的背景，就会导致识别失败或者效果不尽人意；<br>

### 解决方案
[hed-tutorial-for-document-scanning](https://github.com/fengjian0106/hed-tutorial-for-document-scanning)(博客地址这里：[手机端运行卷积神经网络的一次实践 -- 基于 TensorFlow 和 OpenCV 实现文档检测功能](http://fengjian0106.github.io/2017/05/08/Document-Scanning-With-TensorFlow-And-OpenCV/))<br>
这里还有一篇博客参考[Fast and Accurate Document Detection for Scanning](https://blogs.dropbox.com/tech/2016/08/fast-and-accurate-document-detection-for-scanning/)<br>
[hed-tutorial-for-document-scanning](https://github.com/fengjian0106/hed-tutorial-for-document-scanning)只提供了iOS的demo,笔者参考demo，写了Android，在这里提供大家参考。
