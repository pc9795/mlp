package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Linear activation y = f(x)
 **/
public class LinearActivationFn implements ActivationFn {
    public double squash(double input) {
        return input;
    }

    public double squashDerivative(double input) {
        return 1;
    }
}
