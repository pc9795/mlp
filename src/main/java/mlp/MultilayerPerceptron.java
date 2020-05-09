package mlp;

import mlp.activations.Activation;
import mlp.activations.TanhActivation;
import mlp.loss_functions.CrossEntropyLoss;
import mlp.loss_functions.Loss;
import mlp.loss_functions.SquaredErrorLoss;

/**
 * Created By: Prashant Chaubey
 * Created On: 23-03-2020 09:41
 * Purpose: TODO:
 **/
public class MultilayerPerceptron {
    //Number of inputs
    private int ni;
    //Number of units in hidden layer
    private int nh;
    //Number of outputs
    private int no;
    private Activation activation;
    private Loss lossFunction;
    private int epochs;
    private double learningRate;
    //Weights of the lower layer
    private double w1[][];
    //Weights of the upper layer
    private double w2[][];
    //Weight changes going to be applied to lower layer
    private double dw1[][];
    //Weight changes going to be applied to upper layer
    private double dw2[][];
    //Activations of the lower layer
    private double z1[][];
    //Activations of the upper layer
    private double z2[][];
    //Contains value of hidden units
    private double h[];
    //Contains value of the outputs
    private double o[];


    public MultilayerPerceptron(int ni, int nh, int no) {
        this.ni = ni;
        this.nh = nh;
        this.no = no;
        this.activation = new TanhActivation();
        this.lossFunction = no > 1 ? new CrossEntropyLoss() : new SquaredErrorLoss();
        this.epochs = 1000;
        this.learningRate = 0.1;
        this.h = new double[nh];
        this.o = new double[no];
        this.w1 = new double[ni][nh];
        //All values will be zero by default
        this.dw1 = new double[ni][nh];
        this.w2 = new double[nh][no];
        //All values will be zero by default
        this.dw2 = new double[nh][no];
        //Initialize weights to small random values.
        randomise();
    }

    private void randomise() {
        double upperLimit = 0.5;
        double lowerLimit = -0.5;

        for (int i = 0; i < this.w1.length; i++) {
            for (int j = 0; j < this.w1[0].length; j++) {
                this.w1[i][j] = (Math.random() * (upperLimit - lowerLimit)) + lowerLimit;
            }
        }

        for (int i = 0; i < this.w2.length; i++) {
            for (int j = 0; j < this.w2[0].length; j++) {
                this.w2[i][j] = (Math.random() * (upperLimit - lowerLimit)) + lowerLimit;
            }
        }
    }

    private void forward(double[] input) {
        //Forward pass. Input vector I is processed to produce an output, which is stored in O[].
    }

    private double backward(double[] target) {
        //Backwards pass. Target t is compared with output O, deltas are computed for the upper layer, and are
        // multiplied by the inputs to the layer (the values in H) to produce the weight updates which are stored in dW2
        // (added to it, as you may want to store these for many examples). Then deltas are produced for the lower
        // layer, and the same process is repeated here, producing weight updates to be added to dW1.
        // Returns the error on the example.
        return -1;
    }

    private void updateWeights(double learningRate) {
        //this simply does (component by component, i.e. within for loops):
        // W1 += learningRate*dW1;
        // W2 += learningRate*dW2;
        // dW1 = 0;
        // dW2 = 0;
    }

    public void fit(double x[], double y[]) {
        //for (int e=0; e<epochs; e++) {
        // error = 0;
        // for (int p=0; p< numExamples; p++) {
        // NN.forward(example[p].input);
        // error += NN.backwards(example[p].output);
        // every now and then {
        // updateWeights(some_small_value);
        // }
        // }
        // cout << “Error at epoch “ << e << “ is “ << error << “\n”;
        // }
    }

    public double[] predict(double x[]) {
        return null;
    }
}
