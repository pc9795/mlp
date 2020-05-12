package mlp.activations;

import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Linear activation y = f(x)
 **/
public class LinearActivationFn implements ActivationFn {
    @Override
    public double[] squash(double[] input) {
        double[] output = new double[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }

    @Override
    public double[] squashDerivative(double[] input) {
        double[] output = new double[input.length];
        Arrays.fill(output, 1);
        return output;
    }
}
