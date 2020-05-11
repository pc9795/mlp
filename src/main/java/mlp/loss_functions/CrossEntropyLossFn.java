package mlp.loss_functions;

import mlp.exceptions.MLPException;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:28
 * Purpose: Cross entropy loss
 **/
public class CrossEntropyLossFn implements LossFn {
    public double calculate(double[] output, double[] target) {
        if (output.length != target.length) {
            throw new MLPException(String.format("The length of output and target vector is different. %s != %s",
                    output.length, target.length));
        }
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            loss += (-target[i] * Math.log(output[i])) - ((1 - target[i]) * Math.log(1 - output[i]));
        }
        return loss;
    }
}
