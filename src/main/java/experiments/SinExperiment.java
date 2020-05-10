package experiments;

import com.sun.org.apache.xpath.internal.operations.Mult;
import mlp.MultilayerPerceptron;

import javax.rmi.CORBA.Util;
import java.util.Random;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 00:41
 * Purpose: TODO:
 **/
public class SinExperiment {
    public static void main(String[] args) {
        int upperLimit = 1;
        int lowerLimit = -1;
        int vectorSize = 4;
        int count = 200;
        int randomStateInput = 20;
        int randomStateMLP = 20;
        int hiddenUnits = 5;
        double[][] input = generateRandomVectors(randomStateInput, count, vectorSize, lowerLimit, upperLimit);

        double[][] output = new double[count][1];
        for (int i = 0; i < count; i++) {
            double sum = input[i][0];
            boolean positive = false;
            for (int j = 1; j < vectorSize; j++) {
                sum = positive ? sum + input[i][j] : sum - input[i][j];
                positive = !positive;
            }
            output[i][0] = Math.sin(sum);
        }

        double splitSize = 0.75;
        Utils.TrainTestSplit trainTestSplit = Utils.trainTestSplit(input, output, splitSize);

        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length, randomStateMLP);
        mlp.fit(trainTestSplit.trainInput, trainTestSplit.trainOutput);

        System.out.println("Training accuracy:" + Utils.accuracyScore(trainTestSplit.trainOutput,
                mlp.predict(trainTestSplit.trainInput)));

        System.out.println("Testing accuracy:" + Utils.accuracyScore(trainTestSplit.testOutput,
                mlp.predict(trainTestSplit.testInput)));
    }

    private static double[][] generateRandomVectors(int randomState, int count, int vectorSize, int lowerLimit,
                                                    int upperLimit) {
        double[][] vectors = new double[count][vectorSize];
        Random random = new Random(randomState);

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < vectorSize; j++) {
                vectors[i][j] = (random.nextDouble() * (upperLimit - lowerLimit)) + lowerLimit;
            }
        }
        return vectors;
    }
}
