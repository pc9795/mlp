package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Tanh (hyperbolic tangent) activation function y = tanh(x)
 **/
public class TanhActivationFn implements ActivationFn {
    public double squash(double input) {
        return Math.tanh(input);
    }

    public double squashDerivative(double input) {
        return 1 - Math.pow(this.squash(input), 2);
    }
}
