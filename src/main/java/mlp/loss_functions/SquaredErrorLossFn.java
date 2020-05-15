package mlp.loss_functions;

import mlp.exceptions.MLPException;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:28
 * Purpose: Squared error loss
 **/
public class SquaredErrorLossFn implements LossFn {
    public double calculate(double[] predicted, double[] target) {
        //The length of predicted output and target output must be same
        if (predicted.length != target.length) {
            throw new MLPException(String.format("The length of output and target vector is different. %s != %s",
                    predicted.length, target.length));
        }

        double loss = 0;
        for (int i = 0; i < predicted.length; i++) {
            loss += 0.5 * Math.pow(target[i] - predicted[i], 2);
        }

        return loss;
    }
}
