package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: TODO:
 **/
public class SigmoidActivation implements Activation {
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }
}
