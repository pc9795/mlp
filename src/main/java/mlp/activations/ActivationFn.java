package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Parent class for all activation functions
 **/
public interface ActivationFn {

    /**
     * Apply the function to the input
     *
     * @param input input
     * @return value of the activation function after applying the input
     */
    double squash(double input);

    /**
     * Apply the derivative of the function to the input
     *
     * @param input input
     * @return value of the derivative of the activation function after applying the input
     */
    double squashDerivative(double input);
}
