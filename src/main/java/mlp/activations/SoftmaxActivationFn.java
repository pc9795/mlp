package mlp.activations;

import mlp.exceptions.MLPException;

import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 11-05-2020 19:56
 * Purpose: Softmax activation for categorical variables
 **/
public class SoftmaxActivationFn implements ActivationFn {
    @Override
    public double[] squash(double[] input) {
        double[] output = new double[input.length];
        double sum = 0;
        for (double anInput : input) {
            sum += Math.exp(anInput);
        }
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]) / sum;
        }
        return output;
    }

    @Override
    public double[] squashDerivative(double[] input) {
        throw new MLPException("We are not explicitly calculating derivative of the softmax function. A simplification " +
                "is used in calculation of the delta for last layer and softmax is always used for last layer only");
    }
}
