package mlp.loss_functions;

import mlp.exceptions.MLPException;

/**
 * Created By: Prashant Chaubey
 * Created On: 11-05-2020 19:38
 * Purpose: Categorical cross entropy for multi-class classification
 **/
public class CategoricalCrossEntropyLossFn implements LossFn {
    @Override
    public double calculate(double[] predicted, double[] target) {
        if (predicted.length != target.length) {
            throw new MLPException(String.format("The length of output and target vector is different. %s != %s",
                    predicted.length, target.length));
        }
        double loss = 0;
        for (int i = 0; i < target.length; i++) {
            loss += -(target[i] * Math.log(predicted[i]));
        }
        return loss;
    }
}
