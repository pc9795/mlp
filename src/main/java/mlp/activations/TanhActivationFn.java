package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Tanh (hyperbolic tangent) activation function y = tanh(x)
 **/
public class TanhActivationFn implements ActivationFn {
    @Override
    public double[] squash(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = Math.tanh(input[i]);
        }

        return output;
    }

    @Override
    public double[] squashDerivative(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = 1 - Math.pow(Math.tanh(input[i]), 2);
        }

        return output;
    }
}
