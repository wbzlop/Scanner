package com.document.pdf.scanner.filter;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;



/**
 * Created by wubaozeng on 2018/9/13.
 * 动态调整 view 宽高，用于规避多次设置 filter 导致的背景噪点问题
 */

public class GLSurfaceCompatView extends GLSurfaceView {

    private int bitmapWidth = 0;
    private int bitmapHeight = 0;

    public GLSurfaceCompatView(Context context)
    {
        super(context);

    }

    public GLSurfaceCompatView(Context context, AttributeSet attrs) {
        super(context, attrs);

    }

    public void setWH(int width,int height)
    {
        this.bitmapWidth = width;
        this.bitmapHeight = height;
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {

            final int widthSpecSize = MeasureSpec.getSize(widthMeasureSpec);
            final int heightSpecSize = MeasureSpec.getSize(heightMeasureSpec);

            final int dwidth = this.bitmapWidth;
            final int dheight = this.bitmapHeight;

            final int vwidth = widthSpecSize - getPaddingLeft() - getPaddingRight();
            final int vheight = heightSpecSize - getPaddingTop() - getPaddingBottom();

            float scale = Math.min((float) vwidth / (float) dwidth, (float) vheight / (float) dheight);

            float dx = vwidth - dwidth * scale;
            float dy = vheight - dheight * scale;

            int outputWidth = (int) Math.ceil(vwidth - dx);
            int outputHeight = (int) Math.ceil(vheight - dy);

            if (outputWidth != 0 && outputHeight != 0) {
                setMeasuredDimension(outputWidth, outputHeight);
            } else {
                super.onMeasure(widthMeasureSpec, heightMeasureSpec);
            }


    }
}
