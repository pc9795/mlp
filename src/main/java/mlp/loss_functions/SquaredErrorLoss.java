package mlp.loss_functions;

import mlp.exceptions.MLPException;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:28
 * Purpose: TODO:
 **/
public class SquaredErrorLoss implements Loss {
    public double calculate(double[] output, double[] target) {
        if (output.length != target.length) {
            throw new MLPException(String.format("The length of ouput and target vector is different. %s != %s", output.length, target.length));
        }
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            loss += Math.pow(target[i] - output[i], 2);
        }
        loss = Math.sqrt(loss);
        return loss;
    }
}
