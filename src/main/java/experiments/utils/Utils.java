package experiments.utils;

import mlp.exceptions.MLPException;

import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 10-05-2020 14:11
 * Purpose: Utility methods for the experiments
 **/
public class Utils {

    /**
     * Calculate a score based on number of correct predictions
     *
     * @param predicted predicted output
     * @param target    actual output
     * @return ration of correct predictions to total predictions
     */
    public static double accuracyScore(double predicted[][], double target[][]) {
        if (target.length != predicted.length) {
            throw new MLPException(String.format("The length of target and predicted is not same: %s != %s",
                    target.length, predicted.length));
        }
        //Check how many predicted values match with output
        double correct = 0;
        for (int i = 0; i < target.length; i++) {
            correct += Arrays.equals(target[i], predicted[i]) ? 1 : 0;
        }
        //Return the ration of the correctly predicted to total number of predictions
        return correct / target.length;
    }

    /**
     * Class to contains information about training and testing data
     */
    public static class TrainTestSplit {
        public double[][] trainInput; //Input to be used for training
        public double[][] testInput; //Input to be used for testing
        public double[][] trainOutput; //Output to be used for training
        public double[][] testOutput; //Output to be used for testing

        TrainTestSplit(double[][] trainInput, double[][] testInput, double[][] trainOutput,
                       double[][] testOutput) {
            this.trainInput = trainInput;
            this.testInput = testInput;
            this.trainOutput = trainOutput;
            this.testOutput = testOutput;
        }
    }

    /**
     * Split the data into training and testing set
     *
     * @param input     input data
     * @param output    output data
     * @param trainSize size of the training set. Should be between 0 - 1
     * @return the splitted data
     */
    public static TrainTestSplit trainTestSplit(double input[][], double output[][], double trainSize) {
        if (input.length != output.length) {
            throw new MLPException(String.format("The length of input and output is not same %s != %s", input.length,
                    output.length));
        }
        if (trainSize < 0 || trainSize > 1) {
            throw new MLPException("The training size should be between 0(inclusive) and 1(inclusive)");
        }
        int count = input.length;
        int inputVectorSize = input[0].length;
        int outputVectorSize = output[0].length;

        int splitPoint = (int) (count * trainSize);
        double trainInput[][] = new double[splitPoint][inputVectorSize];
        double testInput[][] = new double[count - splitPoint][inputVectorSize];
        double trainOutput[][] = new double[splitPoint][outputVectorSize];
        double testOutput[][] = new double[count - splitPoint][outputVectorSize];

        for (int i = 0; i < input.length; i++) {
            if (i < splitPoint) {
                System.arraycopy(input[i], 0, trainInput[i], 0, inputVectorSize);
                System.arraycopy(output[i], 0, trainOutput[i], 0, outputVectorSize);
            } else {
                System.arraycopy(input[i], 0, testInput[i - splitPoint], 0, inputVectorSize);
                System.arraycopy(output[i], 0, testOutput[i - splitPoint], 0, outputVectorSize);
            }
        }
        return new TrainTestSplit(trainInput, testInput, trainOutput, testOutput);
    }

    public static void prettyPrintPrediction(double[][] predicted, double[][] output) {
        if (output.length != predicted.length) {
            throw new MLPException(String.format("The length of output and predicted is not same: %s != %s",
                    output.length, predicted.length));
        }
        System.out.println("Output;Predicted");
        for (int i = 0; i < output.length; i++) {
            System.out.println(Arrays.toString(output[i]) + ";" + Arrays.toString(predicted[i]));
        }
    }
}