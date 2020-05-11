package mlp.loss_functions;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:28
 * Purpose: Parent class for all loss functions
 **/
public interface LossFn {
    /**
     * Caculate loss for given output and target values. "Target" values signifies actual output and "output" values
     * signifies the values obtained from the multi-layer perceptron.
     *
     * @param output output of the mlp
     * @param target actual output values
     * @return loss between output and target
     */
    double calculate(double[] output, double[] target);
}
