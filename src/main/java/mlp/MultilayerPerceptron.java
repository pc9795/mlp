package mlp;

import mlp.activations.*;
import mlp.exceptions.MLPException;
import mlp.loss_functions.BinaryCrossEntropyLossFn;
import mlp.loss_functions.CategoricalCrossEntropyLossFn;
import mlp.loss_functions.LossFn;
import mlp.loss_functions.SquaredErrorLossFn;

import java.util.Random;

/**
 * Created By: Prashant Chaubey
 * Created On: 23-03-2020 09:41
 * Purpose: Class that implements a multi layer perceptron with a single hidden layer
 **/
@SuppressWarnings("unused")
public class MultilayerPerceptron {
    private int ni; //Number of input units
    private int nh; //Number of hidden layer units
    private int no; //Number of output units
    private ActivationFn hiddenActivationFn; //Activation function used in hidden layers
    private ActivationFn outputActivationFn; //Activation function used in output layers
    private LossFn lossFn; // Function to calculate loss between actual output and output of the mlp
    private int epochs; //Epoch to train the mlp
    private double learningRate; //Learning rate for the weight updates
    private double w1[][]; //Weights of the lower layer
    private double w2[][]; //Weights of the upper layer
    private double dw1[][]; //Weight changes going to be applied to lower layer
    private double dw2[][]; //Weight changes going to be applied to upper layer
    private double z1[]; //Activations of the lower layer
    private double z2[]; //Activations of the upper layer
    private double input[]; //Contains value of input units
    private double h[]; //Contains value of hidden units
    private double o[]; //Contains value of the outputs
    private int randomState; //Random state to control the outcomes of mlp
    private boolean classification; //true if this mlp is solving a classification problem
    private boolean multiClass; //true if this mlp is solving a multi class classification problem. If it is not a
    //classification problem then there is no effect of this value.
    private int batchSize; //Batch size for mini-batch gradient descent. If it is 1 then it is stochastic gradient
    // descent and if it is equal to size of the training data then it is batch gradient descent.

    /**
     * MLP for stochastic gradient descent (batch size = 1)
     *
     * @param ni             units in input layers
     * @param nh             units in hidden layers
     * @param no             units in output layers
     * @param learningRate   learning rate for gradient descent
     * @param epochs         number of epochs to run for training
     * @param type           type of  activation for hidden layers
     * @param classification true if it is a classification problem
     * @param multiClass     true if it is a multi-class classification problem
     */
    public MultilayerPerceptron(int ni, int nh, int no, double learningRate, int epochs, ActivationType type,
                                boolean classification, boolean multiClass) {
        this(ni, nh, no, new Random().nextInt(Integer.MAX_VALUE), learningRate, epochs, type, classification,
                multiClass, 1);
    }

    /**
     * MLP for stochastic gradient descent (batch size = 1). We can pass a random state to make initial weight and bias
     * initialization predictable.
     *
     * @param ni             units in input layers
     * @param nh             units in hidden layers
     * @param no             units in output layers
     * @param randomState    seed for random initialization of weights and biases
     * @param learningRate   learning rate for gradient descent
     * @param epochs         number of epochs to run for training
     * @param type           type of  activation for hidden layers
     * @param classification true if it is a classification problem
     * @param multiClass     true if it is a multi-class classification problem
     */
    public MultilayerPerceptron(int ni, int nh, int no, int randomState, double learningRate, int epochs,
                                ActivationType type, boolean classification, boolean multiClass) {
        this(ni, nh, no, randomState, learningRate, epochs, type, classification, multiClass, 1);
    }

    /**
     * MLP for mini-batch gradient descent.
     *
     * @param ni             units in input layers
     * @param nh             units in hidden layers
     * @param no             units in output layers
     * @param learningRate   learning rate for gradient descent
     * @param epochs         number of epochs to run for training
     * @param type           type of  activation for hidden layers
     * @param classification true if it is a classification problem
     * @param multiClass     true if it is a multi-class classification problem
     * @param batchSize      batch size of gradient descent
     */
    public MultilayerPerceptron(int ni, int nh, int no, double learningRate, int epochs, ActivationType type,
                                boolean classification, boolean multiClass, int batchSize) {
        this(ni, nh, no, new Random().nextInt(Integer.MAX_VALUE), learningRate, epochs, type, classification,
                multiClass, batchSize);
    }

    /**
     * MLP for mini-batch gradient descent.We can pass a random state to make initial weight and bias
     * initialization predictable.
     *
     * @param ni             units in input layers
     * @param nh             units in hidden layers
     * @param no             units in output layers
     * @param randomState    seed for random initialization of weights and biases
     * @param learningRate   learning rate for gradient descent
     * @param epochs         number of epochs to run for training
     * @param type           type of  activation for hidden layers
     * @param classification true if it is a classification problem
     * @param multiClass     true if it is a multi-class classification problem
     * @param batchSize      batch size of gradient descent
     */
    public MultilayerPerceptron(int ni, int nh, int no, int randomState, double learningRate, int epochs,
                                ActivationType type, boolean classification, boolean multiClass, int batchSize) {
        this.ni = ni;
        this.nh = nh;
        this.no = no;
        this.randomState = randomState;
        this.hiddenActivationFn = this.getActivation(type);
        if (!classification) {
            //Regression
            this.outputActivationFn = new LinearActivationFn();
            this.lossFn = new SquaredErrorLossFn();
        } else if (multiClass) {
            //Multi-class classification
            this.outputActivationFn = new SoftmaxActivationFn();
            this.lossFn = new CategoricalCrossEntropyLossFn();
        } else {
            //Multi-label/binary classification
            this.outputActivationFn = new SigmoidActivationFn();
            this.lossFn = new BinaryCrossEntropyLossFn();
        }
        this.classification = classification;
        this.multiClass = multiClass;
        this.batchSize = batchSize;
        this.epochs = epochs;
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

    /**
     * Get activation function from the given type
     *
     * @param type type of the activation function
     * @return activation function
     */
    private ActivationFn getActivation(ActivationType type) {
        switch (type) {
            case RELU:
                return new ReluActivationFn();
            case SIGMOID:
                return new SigmoidActivationFn();
            case LINEAR:
                return new LinearActivationFn();
            case TANH:
                return new TanhActivationFn();
            case LEAKY_RELU:
                return new LeakyReluActivationFn();
        }
        throw new MLPException(String.format("Type: %s is not registered", type));
    }

    /**
     * Method to initialize the weights of the mlp to small random values. This randomization is controlled by
     * variable `randomState`. Same values of it will cause same weight initialization
     * <p>
     * Xavier initialization is used. REF: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
     */
    private void randomise() {
        double factor = 6;
        if (hiddenActivationFn instanceof SigmoidActivationFn) {
            factor = 2;
        }
        //Bound for weights between input and hidden layer
        double initBound = Math.sqrt(factor / (this.ni + this.nh));
        Random random = new Random(this.randomState);
        //Random values for weights of lower layer - between input and hidden layer
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                //Picking a value between (-initBound, initBound).
                this.w1[i][j] = (random.nextDouble() * (initBound - (-initBound))) + (-initBound);
            }
        }
        //Bound for weights between hidden and output layer
        initBound = Math.sqrt(factor / (this.nh + this.no));
        //Random values for weights of upper layer - between hidden and output layer
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                //Picking a value between (-initBound, initBound).
                this.w2[i][j] = (random.nextDouble() * (initBound - (-initBound))) + (-initBound);
            }
        }
    }

    /**
     * Forward pass of the mlp
     *
     * @param input input applied to the mlp
     */
    private void forward(double[] input) {
        if (input.length != this.ni) {
            throw new MLPException(String.format("Expected no of units in input: %s but found %s", this.ni, input.length));
        }
        //Save the input for calculations during back-propagation
        System.arraycopy(input, 0, this.input, 0, ni);
        //Activate lower layer
        for (int i = 0; i < this.nh; i++) {
            //Activation of current hidden unit
            double sum = 0;
            for (int j = 0; j < this.ni; j++) {
                //Multiply the input unit with the synapse (weight) connecting to the current hidden unit
                sum += this.input[j] * this.w1[j][i];
            }
            //Store the input (activation) coming to the current hidden unit
            this.z1[i] = sum;
        }
        //Apply the activation function to the activations of the hidden layer
        this.h = this.hiddenActivationFn.squash(this.z1);
        //Activate upper layer
        for (int i = 0; i < this.no; i++) {
            //Activation of current output unit
            double sum = 0;
            for (int j = 0; j < this.nh; j++) {
                //Multiply the hidden unit with the synapse (weight) connecting to the current output unit
                sum += this.h[j] * this.w2[j][i];
            }
            //Store the input (activation) coming to the current output unit
            this.z2[i] = sum;
        }
        //Apply the activation function to the activations of the output layer
        this.o = this.outputActivationFn.squash(this.z2);
    }

    /**
     * Backward propagation of the mlp
     *
     * @param target target values for mlp
     */
    private void backward(double[] target) {
        if (target.length != this.no) {
            throw new MLPException(String.format("Expected no of units in target: %s but found %s",
                    this.no, target.length));
        }
        //Delta for upper layer - between hidden units and output units
        double[] delta2 = new double[this.no];
        for (int i = 0; i < this.no; i++) {
            //The combinations used in the code: linear activation + squared error loss for regression ,
            //sigmoid/logistic activation + binary cross entropy for binary and multi-label classification, and
            //softmax activation + categorical cross entropy for multi-class classification result in the below delta
            //for last layer after further calculation and simplification
            //Ref: https://www.ics.uci.edu/~pjsadows/notes.pdf
            //NOTE: In actual formula their is a minus (-) sign in front of the delta but while applying the weight
            //changes we will use addition in place of subtraction and the minus sign of learning rate is cancelled by
            //this negative sign.
            delta2[i] = (target[i] - this.o[i]);
        }
        //Weight difference for upper layer
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                this.dw2[i][j] += this.h[i] * delta2[j];
            }
        }
        //Delta for lower layer
        double[] delta1 = new double[this.nh];
        //Derivatives of the activations of the lower layer
        double[] derivatives = this.hiddenActivationFn.squashDerivative(this.z1);
        for (int i = 0; i < this.nh; i++) {
            //Calculating delta for the current hidden unit. Because a hidden unit affects all the outputs we have to
            //consider the contribution of each output unit in the calculation.
            for (int j = 0; j < this.no; j++) {
                delta1[i] += this.w2[i][j] * delta2[j];
            }
            //Delta is computed using multiplication of the error component with the derivative of the activation
            //received by the unit
            delta1[i] *= derivatives[i];
        }
        //Weight difference for lower layer
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                this.dw1[i][j] += this.input[i] * delta1[j];
            }
        }
    }

    /**
     * Function to update the weights for the mlp from with the weight changes
     */
    private void updateWeights() {
        //Update weights in lower layer
        for (int i = 0; i < this.w1.length; i++) {
            for (int j = 0; j < this.w1[0].length; j++) {
                //A positive sign is used because while calculating the delta we left out the minus sign there. So that
                //minus sign cancels the minus sign here.
                this.w1[i][j] += this.learningRate * this.dw1[i][j];
            }
        }
        //Update weights in upper layer
        for (int i = 0; i < this.w2.length; i++) {
            for (int j = 0; j < this.w2[0].length; j++) {
                //A positive sign is used because while calculating the delta we left out the minus sign there. So that
                //minus sign cancels the minus sign here.
                this.w2[i][j] += this.learningRate * this.dw2[i][j];
            }
        }
        //Reset the weight changes to zeroes.
        this.dw1 = new double[ni][nh];
        this.dw2 = new double[nh][no];
    }

    /**
     * The method to train the mlp with a particular input and output
     *
     * @param x input
     * @param y output
     */
    public void fit(double x[][], double y[][]) {
        System.out.println("Epoch;Loss");
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            //Start of an epoch
            double error = 0;
            //For each example
            for (int i = 0; i < x.length; i++) {
                //Do a forward pass
                this.forward(x[i]);
                //Calculate the error
                error += this.lossFn.calculate(this.o, y[i]);
                //Calculate the weight updates using back-propagation
                this.backward(y[i]);
                //Update the weights with the changes
                //`i` is zero indexed therefore adding 1.
                //Update the weights at the batch end and at the last index
                if ((i + 1) % batchSize == 0 || i == x.length - 1) {
                    updateWeights();
                }
            }
            System.out.println(String.format("%s;%s", epoch, error));
        }
    }

    /**
     * Predict the output for given unit using mlp
     *
     * @param x input
     * @return predicted output
     */
    public double[][] predict(double x[][]) {
        double[][] output = new double[x.length][this.no];
        for (int i = 0; i < x.length; i++) {
            //Do a forward pass
            this.forward(x[i]);
            //Copy the outputs
            System.arraycopy(this.o, 0, output[i], 0, this.no);
        }
        //Apply threshold if it is a binary/multi-label classification problem
        if (this.classification && !this.multiClass) {
            //Threshold for classification problems
            double threshold = 0.5;
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[0].length; j++) {
                    output[i][j] = output[i][j] > threshold ? 1 : 0;
                }
            }
        }
        return output;
    }

    /**
     * Calculate loss for particular prediction and target valuesF
     *
     * @param predicted values predicted
     * @param target    target values
     * @return loss
     */
    public double loss(double predicted[][], double[] target[]) {
        if (target.length != predicted.length) {
            throw new MLPException(String.format("The length of target and predicted is not same: %s != %s",
                    target.length, predicted.length));
        }
        double loss = 0;
        for (int i = 0; i < target.length; i++) {
            loss += this.lossFn.calculate(predicted[i], target[i]);
        }
        return loss / target.length;
    }

    public void printInfo() {
        System.out.println("Weights of lower layer");
        for (int i = 0; i < ni; i++) {
            System.out.print("| ");
            for (int j = 0; j < nh; j++) {
                System.out.print(String.format("%4.5f | ", this.w1[i][j]));
            }
            System.out.println();
        }
        System.out.println("weights of upper layer");
        for (int i = 0; i < nh; i++) {
            System.out.print("| ");
            for (int j = 0; j < no; j++) {
                System.out.print(String.format("%4.5f | ", this.w2[i][j]));
            }
            System.out.println();
        }
    }
}
