package mlp.loss_functions;

import mlp.exceptions.MLPException;

/**
 * Created By: Prashant Chaubey
 * Created On: 11-05-2020 19:53
 * Purpose: Binary cross entropy for multi-label and binary classification
 **/
public class BinaryCrossEntropyLossFn implements LossFn {
    @Override
    public double calculate(double[] predicted, double[] target) {
        //The length of predicted output and target output must be same
        if (predicted.length != target.length) {
            throw new MLPException(String.format("The length of output and target vector is different. %s != %s",
                    predicted.length, target.length));
        }

        double loss = 0;
        for (int i = 0; i < target.length; i++) {
            loss += -(target[i] * Math.log(predicted[i])) - ((1 - target[i]) * Math.log(1 - predicted[i]));
        }

        return loss;
    }
}
