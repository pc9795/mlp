package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: TODO:
 **/
public interface Activation {

    double apply(double input);

    double applyDerivative(double input);
}
