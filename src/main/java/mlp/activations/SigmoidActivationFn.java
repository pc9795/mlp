package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Sigmoid activation function y = 1/(1 + e^-x)
 **/
public class SigmoidActivationFn implements ActivationFn {
    public double squash(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    public double squashDerivative(double input) {
        return this.squash(input) * (1 - this.squash(input));
    }
}
