package experiments;

import mlp.MultilayerPerceptron;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 00:41
 * Purpose: TODO:
 **/
public class XORExperiment {
    public static void main(String[] args) {
        int randomState = 20;
        int hiddenUnits = 4;
        double[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] output = {{0}, {1}, {1}, {0}};

        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length, 20);
        mlp.fit(input, output);

        double predicted[][] = mlp.predict(input);
        System.out.println("Accuracy:" + Utils.accuracyScore(output, predicted));
    }
}
