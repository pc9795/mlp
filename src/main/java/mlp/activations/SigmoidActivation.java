package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: TODO:
 **/
public class SigmoidActivation implements Activation {
    public double apply(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    public double applyDerivative(double input) {
        throw new RuntimeException("Not implemented");
    }
}
