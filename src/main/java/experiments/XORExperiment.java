package experiments;

import experiments.utils.Utils;
import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;
import mlp.exceptions.MLPException;

import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 00:41
 * Purpose: Experiment to test MLP is able to learn xor or not
 **/
public class XORExperiment {
    public static void main(String[] args) {
        //Hyper parameters
        int randomState = 20;
        int hiddenUnits = 4;
        double learningRate = 0.01;
        int epochs = 5000;
        ActivationType type = ActivationType.TANH;

        //Input and output. XOR function on 2 binary variables
        double[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] output = {{0}, {1}, {1}, {0}};

        //Multi layer perceptron object for binary classification problem
        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length,
                randomState, learningRate, epochs, type, true, false);

        //Training
        mlp.fit(input, output);

        //Print the predictions of the MLP
        double predicted[][] = mlp.predict(input);
        System.out.println();
        System.out.println("***********************");
        Utils.prettyPrintPrediction(predicted, output);
        System.out.println("***********************");

        //Print the accuracy and loss of the binary-classification
        System.out.println();
        System.out.println("***********************");
        System.out.println("Loss: " + mlp.loss(predicted, output));
        //The probabilities are converted into binary labels (0 or 1) on the basis of a threshold. The threshold is
        //set to 0.5 currently. A value > 0.5 will be labeled 1 and 0 otherwise.
        applyThresholdInplace(predicted);
        System.out.println("Accuracy: " + accuracyScore(predicted, output));
        System.out.println("***********************");
        System.out.println();

        //MLP info
        System.out.println();
        mlp.printInfo(true);
    }

    /**
     * Get accuracy from the predicted values and the target values. In case of binary classification the number of
     * columns would be 1.
     *
     * @param predicted predicted values
     * @param target    target values
     * @return ration of values correctly predicted.
     */
    private static double accuracyScore(double[][] predicted, double[][] target) {
        //Check the lengths are same
        if (target.length != predicted.length) {
            throw new MLPException(String.format("The length of target and predicted is not same: %s != %s",
                    target.length, predicted.length));
        }

        //Check how many values are correctly predicted.
        double correct = 0;
        for (int i = 0; i < predicted.length; i++) {
            correct += Arrays.equals(predicted[i], target[i]) ? 1 : 0;
        }

        //Return accuracy score as the ration of values correctly predicted.
        return correct / predicted.length;
    }

    /**
     * Labels values as 0 or 1 on the basis of prediction probabilities and a threshold
     *
     * @param predicted prediction probabilities of output
     */
    private static void applyThresholdInplace(double[][] predicted) {
        double threshold = 0.5;

        for (int i = 0; i < predicted.length; i++) {
            for (int j = 0; j < predicted[0].length; j++) {
                //Values which are above a threshold are labeled 1 and 0 other wise
                predicted[i][j] = predicted[i][j] > threshold ? 1 : 0;
            }
        }
    }
}
