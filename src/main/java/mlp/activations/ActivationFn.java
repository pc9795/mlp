package mlp.activations;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:21
 * Purpose: Parent class for all activation functions
 **/
public interface ActivationFn {

    /**
     * Apply the function to the inputs
     *
     * @param x inputs
     * @return value of the inputs after applying the activation function
     */
    double[] squash(double[] x);

    /**
     * Apply the derivative of the function to the inputs
     *
     * @param x inputs
     * @return value of the inputs after applying the derivative of the activation function
     */
    double[] squashDerivative(double[] x);
}
