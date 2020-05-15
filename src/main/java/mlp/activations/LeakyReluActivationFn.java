package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 12-05-2020 00:44
 * Purpose: Leaky Relu activation y = max(0.01x, x)
 **/
public class LeakyReluActivationFn implements ActivationFn {
    private double threshold = 0.01;

    @Override
    public double[] squash(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(this.threshold * input[i], input[i]);
        }

        return output;
    }

    @Override
    public double[] squashDerivative(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] <= 0 ? this.threshold : 1;
        }

        return output;
    }
}
