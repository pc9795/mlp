package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 21:13
 * Purpose: Relu activation y = max(0, x)
 **/
public class ReluActivationFn implements ActivationFn {
    public double squash(double input) {
        return Math.max(0, input);
    }

    public double squashDerivative(double input) {
        return input <= 0 ? 0 : 1;
    }
}
