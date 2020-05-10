package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: TODO:
 **/
public class LinearActivation implements Activation {
    public double apply(double input) {
        return input;
    }

    public double applyDerivative(double input) {
        throw new RuntimeException("Not implemented");
    }
}
