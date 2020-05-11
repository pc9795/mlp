package experiments;

import experiments.utils.Utils;
import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;

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
        int hiddenUnits = 10;
        double learningRate = 0.1;
        int epochs = 500;
        ActivationType type = ActivationType.LINEAR;
        //Input and output
        int count = 20000;
        int outputVectorSize = 26;
        int inputVectorSize = 16;
        double[][] input = new double[count][inputVectorSize];
        double[][] output = new double[count][outputVectorSize];
        Scanner in = new Scanner(new FileInputStream(new File("letter-recognition.data")));
        for (int i = 0; i < count; i++) {
            String line = in.nextLine();
            String[] tokens = line.split(",");
            if (tokens.length != inputVectorSize + 1) {
                throw new RuntimeException("Invalid input");
            }
            //Get the character
            char target = tokens[0].charAt(0);
            output[i][target - 'A'] = 1;
            //Get the input
            for (int j = 1; j < tokens.length; j++) {
                input[i][j - 1] = Double.parseDouble(tokens[j]);
            }
        }
        //Split the data into 80:20 ratio for training and testing respectively.
        double splitSize = 0.8;
        Utils.TrainTestSplit trainTestSplit = Utils.trainTestSplit(input, output, splitSize);
        //Multilayer perceptron object
        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length,
                randomState, learningRate, epochs, type, true, true);
        //Training
        mlp.fit(trainTestSplit.trainInput, trainTestSplit.trainOutput);
        //Prediction
        System.out.println("Training accuracy:" + Utils.accuracyScore(mlp.predict(trainTestSplit.trainInput), trainTestSplit.trainOutput));
        System.out.println("Training loss:" + mlp.loss(mlp.predict(trainTestSplit.trainInput), trainTestSplit.trainOutput));
        System.out.println("Testing accuracy:" + Utils.accuracyScore(mlp.predict(trainTestSplit.testInput), trainTestSplit.testOutput));
        System.out.println("Testing loss:" + mlp.loss(mlp.predict(trainTestSplit.testInput), trainTestSplit.testOutput));
        in.close();
    }
}
