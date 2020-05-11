package experiments;

import experiments.utils.Utils;
import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;

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
        double learningRate = 0.1;
        int epochs = 500;
        ActivationType type = ActivationType.LINEAR;
        //Input and output
        double[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] output = {{0}, {1}, {1}, {0}};
        //Multi layer perceptron object
        MultilayerPerceptron mlp = new MultilayerPerceptron(input[0].length, hiddenUnits, output[0].length,
                randomState, learningRate, epochs, type);
        //Training
        mlp.fit(input, output);
        //Prediction
        double predicted[][] = mlp.predict(input);
        System.out.println("***********************");
        Utils.prettyPrintPrediction(predicted, output);
        System.out.println("***********************");
        System.out.println("Accuracy:" + mlp.accuracyScore(predicted, output));
        System.out.println("***********************");
        mlp.printInfo();
    }
}
