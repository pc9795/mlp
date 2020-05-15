package experiments;

import experiments.utils.Utils;
import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;
import mlp.exceptions.MLPException;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 00:42
 * Purpose: Experiment to check mlp can classify letter recognition data-set or not.
 * (http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data)
 **/
public class LetterRecognitionExperiment {
    public static void main(String[] args) throws FileNotFoundException {
        //Hyper parameters
        int randomState = 20;
        int hiddenUnits = 150;
        double learningRate = 0.05;
        int epochs = 1000;
        ActivationType type = ActivationType.RELU;
        int batchSize = 50;

        //Input and output
        int count = 20000;
        int outputVectorSize = 26;
        int inputVectorSize = 16;
        double[][] input = new double[count][inputVectorSize];
        double[][] output = new double[count][outputVectorSize];
        String filePath = "letter-recognition.data"; //The file is at the root of the project currently. Update the path
        // here accordingly if changed.

        Scanner in = new Scanner(new FileInputStream(new File(filePath)));
        for (int i = 0; i < count; i++) {
            String line = in.nextLine(); //Read a line
            String[] tokens = line.split(","); //Split the line by commas

            //Validation. +1 because of the target alphabet
            if (tokens.length != inputVectorSize + 1) {
                throw new RuntimeException("Invalid input");
            }

            //Get the character
            char target = tokens[0].charAt(0); //Get the target alphabet
            output[i][target - 'A'] = 1; // Mark the position represented by the alphabet (0-25) with 1.

            //Get the input
            for (int j = 1; j < tokens.length; j++) {
                input[i][j - 1] = Double.parseDouble(tokens[j]);
            }
        }
        in.close();

        //Split the data into 80:20 ratio for training and testing respectively.
        double splitSize = 0.8;
        Utils.TrainTestSplit trainTestSplit = Utils.trainTestSplit(input, output, splitSize);

        //Normalization - Using min-max scaling. The values will lie in the range [0, 1] after scaling.
        Utils.MinMaxScaler scaler = new Utils.MinMaxScaler();
        scaler.fit(trainTestSplit.trainInput);
        scaler.inplaceTransform(trainTestSplit.trainInput);
        scaler.inplaceTransform(trainTestSplit.testInput);

        //Multilayer perceptron object - Multi-class classification problem
        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length,
                randomState, learningRate, epochs, type, true, true, batchSize);

        //Training
        long now = System.currentTimeMillis();
        mlp.fit(trainTestSplit.trainInput, trainTestSplit.trainOutput);

        //Print the predictions of the test set
        System.out.println();
        System.out.println("***********************");
        double[][] predicted = mlp.predict(trainTestSplit.testInput);
        prettyPrintPrediction(predicted, trainTestSplit.testOutput);
        System.out.println("***********************");
        System.out.println();

        //Accuracy and loss on training set
        System.out.println("***********************");
        predicted = mlp.predict(trainTestSplit.trainInput);
        System.out.println("Training accuracy:" + accuracyScore(predicted, trainTestSplit.trainOutput));
        System.out.println("Training loss:" + mlp.loss(predicted, trainTestSplit.trainOutput));

        //Accuracy and loss on testing set
        predicted = mlp.predict(trainTestSplit.testInput);
        System.out.println("Testing accuracy:" + accuracyScore(predicted, trainTestSplit.testOutput));
        System.out.println("Testing loss:" + mlp.loss(predicted, trainTestSplit.testOutput));
        System.out.println("Execution time(in secs):" + (System.currentTimeMillis() - now) / 1000.0);
        System.out.println("***********************");
        System.out.println();

        //MLP info
        mlp.printInfo(false);
    }

    /**
     * Calculate a score based on number of correct predictions
     *
     * @param predicted predicted output
     * @param target    actual output
     * @return ration of correct predictions to total predictions
     */
    private static double accuracyScore(double predicted[][], double target[][]) {
        //The length of predicted output and target output must be same.
        if (target.length != predicted.length) {
            throw new MLPException(String.format("The length of target and predicted is not same: %s != %s",
                    target.length, predicted.length));
        }
        //Check how many predicted values match with output
        double correct = 0;
        for (int i = 0; i < target.length; i++) {
            correct += getAlphabet(target[i]) == getAlphabet(predicted[i]) ? 1 : 0;
        }

        //Return the ratio of the correctly predicted to total number of predictions
        return correct / target.length;
    }

    /**
     * Print the predictions of the letter-recognition problem
     *
     * @param predicted predicted output
     * @param output    actual output
     */
    private static void prettyPrintPrediction(double[][] predicted, double[][] output) {
        //THe length of predicted output and target output must be same
        if (output.length != predicted.length) {
            throw new MLPException(String.format("The length of output and predicted is not same: %s != %s",
                    output.length, predicted.length));
        }

        System.out.println("Output;Predicted");
        for (int i = 0; i < output.length; i++) {
            System.out.println(getAlphabet(output[i]) + ";" + getAlphabet(predicted[i]));
        }
    }

    /**
     * Get an alphabet from its softmax probabilities
     *
     * @param probabilities softmax probabilities
     * @return alphabet represented by the position with maximum probability
     */
    private static char getAlphabet(double[] probabilities) {
        int maxIndex = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }

        // 'A' is represented by 0, 'B' by 1 and ...
        return (char) ('A' + maxIndex);
    }
}
