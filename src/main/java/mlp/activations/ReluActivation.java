package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 21:13
 * Purpose: TODO:
 **/
public class ReluActivation implements Activation {
    public double squash(double input) {
        return Math.max(0, input);
    }

    public double squashDerivative(double input) {
        throw new RuntimeException("Not implemented");
    }
}
