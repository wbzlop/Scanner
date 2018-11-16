package com.document.pdf.scanner.hed;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.Operation;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import io.reactivex.Observable;
import io.reactivex.ObservableEmitter;
import io.reactivex.ObservableOnSubscribe;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.functions.Consumer;
import io.reactivex.schedulers.Schedulers;
import me.pqpo.smartcropperlib.SmartCropper;


/**
 * Created by wubaozeng on 2018/10/31.
 * https:github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java#L107
 * https://github.com/fengjian0106/hed-tutorial-for-document-scanning/blob/ddd65cb3f6c9fcb5703bca2442ed39948468ad4e/ios_demo/DemoWithStaticLib/DemoWithStaticLib/ViewController.mm#L183
 */

public class HEDDetect {

    public interface HEDDetectAction {
        void complete(float[] output);
    }


    private final String modelName = "hed_graph.pb";

    private final String inputName = "hed_input";
    private final String outputName = "hed/dsn_fuse/conv2d/BiasAdd";
    private final String isTrainingName = "is_training";

    private int inputSize = 256;

    private int[] intValues = new int[inputSize * inputSize];
    private float[] floatValues = new float[inputSize * inputSize * 3];
    ;
    private float[] outputs;
    private String[] outputNames = new String[]{outputName};
    private boolean[] trainings = new boolean[]{false};

    private boolean logStats = false;

    TensorFlowInferenceInterface inferenceInterface;

    private Context context;

    public HEDDetect(Context context) {
        this.context = context;

        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), modelName);
        final Operation operation = inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        outputs = new float[numClasses * numClasses];
    }

    public void run(Bitmap bitmap, HEDDetectAction action) {
        Observable.create(new ObservableOnSubscribe<float[]>() {
            @Override
            public void subscribe(ObservableEmitter<float[]> emitter) throws Exception {
                float[] output = run(bitmap);
                emitter.onNext(output);
            }
        }).subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(new Consumer<float[]>() {
                    @Override
                    public void accept(float[] floats) throws Exception {
                        if (action != null) {
                            action.complete(floats);
                        }
                    }
                });
    }

    public float[] run(Bitmap bitmap) {

        floatValues = SmartCropper.bitmapToFloats(bitmap);

        //It's no training
        inferenceInterface.feed(isTrainingName, trainings);

//         Copy the input data into TensorFlow.
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);

//         Run the inference call.
        inferenceInterface.run(outputNames, logStats);

//         Copy the output Tensor back into the output array.
        inferenceInterface.fetch(outputName, outputs);


        return outputs;
    }


}
