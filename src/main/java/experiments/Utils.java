package experiments;

import mlp.exceptions.MLPException;

import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 10-05-2020 14:11
 * Purpose: TODO:
 **/
public class Utils {
    public static double accuracyScore(double output[][], double predicted[][]) {
        if (output.length != predicted.length) {
            throw new MLPException(String.format("The length of output and predicted is not same: %s != %s",
                    output.length, predicted.length));
        }
        double correct = 0;
        for (int i = 0; i < output.length; i++) {
            correct += Arrays.equals(output[i], predicted[i]) ? 1 : 0;
        }

        return correct / output.length;
    }

    public static class TrainTestSplit {
        public double[][] trainInput;
        public double[][] testInput;
        public double[][] trainOutput;
        public double[][] testOutput;

        public TrainTestSplit(double[][] trainInput, double[][] testInput, double[][] trainOutput,
                              double[][] testOutput) {
            this.trainInput = trainInput;
            this.testInput = testInput;
            this.trainOutput = trainOutput;
            this.testOutput = testOutput;
        }
    }

    public static TrainTestSplit trainTestSplit(double input[][], double output[][], double trainSize) {
        if (input.length != output.length) {
            throw new MLPException(String.format("The length of input and output is not same %s != %s", input.length,
                    output.length));
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
                System.arraycopy(input[i], 0, testInput[i], 0, inputVectorSize);
                System.arraycopy(output[i], 0, testOutput[i], 0, outputVectorSize);
            }
        }
        return new TrainTestSplit(trainInput, testInput, trainOutput, testOutput);
    }
}
