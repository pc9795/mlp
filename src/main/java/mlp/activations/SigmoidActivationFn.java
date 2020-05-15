package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Sigmoid activation function y = 1/(1 + e^-x)
 **/
public class SigmoidActivationFn implements ActivationFn {
    /**
     * Implementation of sigmoid function
     *
     * @param x input
     * @return output of the sigmoid function with given input
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double[] squash(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = this.sigmoid(input[i]);
        }

        return output;
    }

    @Override
    public double[] squashDerivative(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = this.sigmoid(input[i]) * (1 - this.sigmoid(input[i]));
        }

        return output;
    }
}
