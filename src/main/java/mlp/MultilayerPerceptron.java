package mlp;

import mlp.activations.Activation;
import mlp.activations.SigmoidActivation;
import mlp.exceptions.MLPException;
import mlp.loss_functions.Loss;
import mlp.loss_functions.SquaredErrorLoss;

import java.util.Random;

/**
 * Created By: Prashant Chaubey
 * Created On: 23-03-2020 09:41
 * Purpose: TODO:
 **/
public class MultilayerPerceptron {
    //Default values
    public static final int DEFAULT_EPOCHS = 1000;
    public static final double DEFAULT_LEARNING_STATE = 1e-3;
    public static final double DEFAULT_MOMENTUM = 0.9;
    public static final double DEFAULT_LAMBDA = 0;
    public static final double DEFAULT_EPSILON = 1e-3;

    private int ni; //Number of input units
    private int nh; //Number of units in hidden layer
    private int no; //Number of output units
    private Activation activation;
    private Loss lossFunction;
    private int epochs;
    private double learningRate;
    private double w1[][]; //Weights of the lower layer
    private double w2[][]; //Weights of the upper layer
    private double dw1[][]; //Weight changes going to be applied to lower layer
    private double dw2[][]; //Weight changes going to be applied to upper layer
    private double z1[]; //Activations of the lower layer
    private double z2[]; //Activations of the upper layer
    private double input[]; //Contains value of input units
    private double h[]; //Contains value of hidden units
    private double o[]; //Contains value of the outputs
    private int randomState;


    public MultilayerPerceptron(int ni, int nh, int no, int randomState) {
        this(ni, nh, no, randomState, DEFAULT_LEARNING_STATE);
    }

    public MultilayerPerceptron(int ni, int nh, int no, int randomState, double learningRate) {
        this.ni = ni;
        this.nh = nh;
        this.no = no;
        this.randomState = randomState;
        this.activation = new SigmoidActivation();
        this.lossFunction = new SquaredErrorLoss();
        this.epochs = DEFAULT_EPOCHS;
        this.learningRate = learningRate;
        this.input = new double[ni];
        this.h = new double[nh];
        this.z1 = new double[nh];
        this.o = new double[no];
        this.z2 = new double[no];
        this.w1 = new double[ni][nh];
        this.dw1 = new double[ni][nh]; //All values will be zero by default
        this.w2 = new double[nh][no];
        this.dw2 = new double[nh][no]; //All values will be zero by default
        randomise();
    }

    private void randomise() {
        double upperLimit = 0.5;
        double lowerLimit = -0.5;
        Random random = new Random(this.randomState);

        //Random values for weights of lower layer - between input and hidden layer
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                this.w1[i][j] = (random.nextDouble() * (upperLimit - lowerLimit)) + lowerLimit;
            }
        }
        //Random values for weights of upper layer - between hidden and output layer
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                this.w2[i][j] = (random.nextDouble() * (upperLimit - lowerLimit)) + lowerLimit;
            }
        }
    }

    private void forward(double[] input) {
        if (input.length != this.ni) {
            throw new MLPException(String.format("Expected no of units in input: %s but found %s", this.ni, input.length));
        }
        System.arraycopy(input, 0, this.input, 0, ni);
        //Activate hidden layer
        for (int i = 0; i < this.nh; i++) {
            double sum = 0;
            for (int j = 0; j < this.ni; j++) {
                sum += this.input[j] * this.w1[j][i];
            }
            this.z1[i] = sum;
            this.h[i] = this.activation.squash(sum);
        }
        //Activate output layer
        for (int i = 0; i < this.no; i++) {
            double sum = 0;
            for (int j = 0; j < this.nh; j++) {
                sum += this.h[j] * this.w2[j][i];
            }
            this.z2[i] = sum;
            this.o[i] = this.activation.squash(sum);
        }
    }

    private void backward(double[] target) {
        if (target.length != this.no) {
            throw new MLPException(String.format("Expected no of units in target: %s but found %s",
                    this.no, target.length));
        }
        //Delta for upper layer
        double[] delta2 = new double[this.no];
        for (int i = 0; i < this.no; i++) {
            delta2[i] = (target[i] - this.o[i]) * this.activation.squashDerivative(this.z2[i]);
        }
        //Weight difference for upper layer
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                this.dw2[i][j] = this.h[i] * delta2[j];
            }
        }
        //Delta for lower layer
        double[] delta1 = new double[this.nh];
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                delta1[i] += this.w2[i][j] * delta2[j];
            }
            delta1[i] *= this.activation.squashDerivative(this.z1[i]);
        }
        //Weight difference for lower layer
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                this.dw1[i][j] = this.input[i] * delta1[j];
            }
        }
    }

    private void updateWeights() {
        //Update weights in lower layer
        for (int i = 0; i < this.w1.length; i++) {
            for (int j = 0; j < this.w1[0].length; j++) {
                this.w1[i][j] += this.learningRate * this.dw1[i][j];
            }
        }
        //Update weights in upper layer
        for (int i = 0; i < this.w2.length; i++) {
            for (int j = 0; j < this.w2[0].length; j++) {
                this.w2[i][j] += this.learningRate * this.dw2[i][j];
            }
        }
        //Reset the deltas.
        this.dw1 = new double[ni][nh];
        this.dw2 = new double[nh][no];
    }

    public void fit(double x[][], double y[][]) {
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            double error = 0;
            for (int i = 0; i < x.length; i++) {
                this.forward(x[i]);
                error += this.lossFunction.calculate(this.o, y[i]);
                this.backward(y[i]);
                updateWeights();
            }
            System.out.println(String.format("Error at epoch: %s is %s", epoch, error));
        }
    }

    public double[][] predict(double x[][]) {
        double[][] output = new double[x.length][this.no];
        for (int i = 0; i < x.length; i++) {
            this.forward(x[i]);
            System.arraycopy(this.o, 0, output[i], 0, this.no);
        }
        return output;
    }
}
