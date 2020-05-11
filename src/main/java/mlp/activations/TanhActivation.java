package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: TODO:
 **/
public class TanhActivation implements Activation {
    public double squash(double input) {
        return Math.tanh(input);
    }

    public double squashDerivative(double input) {
        return 1 - Math.pow(Math.tanh(input), 2);
    }
}
