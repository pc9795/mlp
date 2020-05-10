package experiments;

import mlp.MultilayerPerceptron;

import javax.rmi.CORBA.Util;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 00:42
 * Purpose: TODO:
 **/
public class LetterRecognitionExperiment {
    public static void main(String[] args) throws FileNotFoundException {
        int count = 20000;
        int outputVectorSize = 26;
        int inputVectorSize = 17;
        int randomState = 20;
        int hiddenUnits = 10;
        double[][] input = new double[count][inputVectorSize];
        double[][] output = new double[count][outputVectorSize];
        Scanner in = new Scanner(new FileInputStream(new File("letter-recognition.data")));
        for (int i = 0; i < count; i++) {
            String line = in.nextLine();
            String[] tokens = line.split(",");
            if (tokens.length != inputVectorSize + 1) {
                throw new RuntimeException("Invalid input");
            }

            char target = tokens[0].charAt(0);
            output[i][target - 'A'] = 1;
            for (int j = 1; j < tokens.length; j++) {
                input[i][j - 1] = Double.parseDouble(tokens[j]);
            }
        }

        double splitSize = 0.8;
        Utils.TrainTestSplit trainTestSplit = Utils.trainTestSplit(input, output, splitSize);

        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length, randomState);
        mlp.fit(trainTestSplit.trainInput, trainTestSplit.trainOutput);

        System.out.println("Training accuracy:" + Utils.accuracyScore(trainTestSplit.trainOutput,
                mlp.predict(trainTestSplit.trainInput)));

        System.out.println("Testing accuracy:" + Utils.accuracyScore(trainTestSplit.testOutput,
                mlp.predict(trainTestSplit.testInput)));
        in.close();
    }
}
