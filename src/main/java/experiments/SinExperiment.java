package experiments;

import experiments.utils.Utils;
import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;

import java.util.Random;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 00:41
 * Purpose: Experiment to test the MLP can learn a sin function or not
 **/
public class SinExperiment {
    public static void main(String[] args) {
        //Hyper parameters
        int randomStateMLP = 20;
        int hiddenUnits = 5;
        double learningRate = 0.1;
        int epochs = 500;
        ActivationType type = ActivationType.TANH;

        //Input and output
        int upperLimit = 1;
        int lowerLimit = -1;
        int vectorSize = 4;
        int count = 200;
        int randomStateInput = 20;
        //Generate input vectors
        double[][] input = generateRandomVectors(randomStateInput, count, vectorSize, lowerLimit, upperLimit);
        //Generate output
        double[][] output = new double[count][1];
        for (int i = 0; i < count; i++) {
            //We know the vector size is 4 that's why used this type of representation. This line of code is not
            //extensible if we want to increase the vector size but it is more readable.
            output[i][0] = Math.sin(input[i][0] - input[i][1] + input[i][2] - input[i][3]);
        }

        //Splitting the data into 150 training examples and 50 testing examples
        double splitSize = 0.75;
        Utils.TrainTestSplit trainTestSplit = Utils.trainTestSplit(input, output, splitSize);

        //Multilayer perceptron object for regression problem
        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length,
                randomStateMLP, learningRate, epochs, type, false, false);

        //Training
        mlp.fit(trainTestSplit.trainInput, trainTestSplit.trainOutput);

        //Print the predictions of the MLP
        System.out.println();
        System.out.println("***********************");
        double predicted[][] = mlp.predict(trainTestSplit.testInput);
        Utils.prettyPrintPrediction(predicted, trainTestSplit.testOutput);
        System.out.println("***********************");
        System.out.println();

        //Loss on training and test set.
        System.out.println("***********************");
        System.out.println("Training loss: " + mlp.loss(mlp.predict(trainTestSplit.trainInput), trainTestSplit.trainOutput));
        System.out.println("Testing loss: " + mlp.loss(predicted, trainTestSplit.testOutput));
        System.out.println("***********************");
        System.out.println();

        //MLP info
        mlp.printInfo(true);
    }

    /**
     * Generate random vectors
     *
     * @param randomState seed to make vectors deterministic
     * @param count       number of vectors to generate
     * @param vectorSize  size of the vector
     * @param lowerLimit  lower limit of the vector value
     * @param upperLimit  upper limit of the vector value
     * @return generated vectors
     */
    private static double[][] generateRandomVectors(int randomState, int count, int vectorSize, int lowerLimit,
                                                    int upperLimit) {
        double[][] vectors = new double[count][vectorSize];
        Random random = new Random(randomState);

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < vectorSize; j++) {
                //Generate a random value in the range of (lowerLimit, upperLimit)
                vectors[i][j] = (random.nextDouble() * (upperLimit - lowerLimit)) + lowerLimit;
            }
        }

        return vectors;
    }
}
