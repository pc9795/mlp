package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 21:13
 * Purpose: Relu activation y = max(0, x)
 **/
public class ReluActivationFn implements ActivationFn {
    @Override
    public double[] squash(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    @Override
    public double[] squashDerivative(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] <= 0 ? 0 : 1;
        }
        return output;
    }
}
