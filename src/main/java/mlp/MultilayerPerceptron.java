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
    private LossFn lossFn; // Function to calculate loss between actual output and mlp predictions
    private int epochs; //Epochs to train the mlp
    private double learningRate; //Learning rate for the weight updates
    private double w1[][]; //Weights of the lower layer
    private double w2[][]; //Weights of the upper layer
    private double dw1[][]; //Weight changes going to be applied to lower layer
    private double dw2[][]; //Weight changes going to be applied to upper layer
    private double b1[]; //Biases of the lower layer
    private double b2[]; //Biases of the upper layer
    private double db1[]; //Bias changes going to be applied to lower layer
    private double db2[]; //Bias changes going to be applied to upper layer
    private double z1[]; //Activations of the lower layer
    private double z2[]; //Activations of the upper layer
    private double input[]; //Contains value of input units
    private double h[]; //Contains value of hidden units
    private double o[]; //Contains value of the outputs
    private int randomState; //Random state to control the outcomes of mlp
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

        //Choosing loss function and outer layer activation according to the problem (regression, binary/multi-label
        //classification, multi-class classification)
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
        this.b1 = new double[nh];
        this.db1 = new double[nh]; //All values will be zero by default
        this.b2 = new double[no];
        this.db2 = new double[no]; //All values will be zero by default
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
        throw new MLPException(String.format("Type: %s cannot be applied as an activation for hidden layers", type));
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

        //Bound for weights between input and hidden layer. Xavier Initialization
        double initBound = Math.sqrt(factor / (this.ni + this.nh));
        Random random = new Random(this.randomState);

        //Random values for weights of lower layer - between input and hidden layer
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                //Picking a value between (-initBound, initBound).
                this.w1[i][j] = (random.nextDouble() * (initBound - (-initBound))) + (-initBound);
            }
        }

        //Random values for biases of lower layer - between input and hidden layer
        for (int i = 0; i < this.nh; i++) {
            //Picking a value between (-initBound, initBound).
            this.b1[i] = (random.nextDouble() * (initBound - (-initBound))) + (-initBound);
        }

        //Bound for weights between hidden and output layer. Xavier Initialization
        initBound = Math.sqrt(factor / (this.nh + this.no));

        //Random values for weights of upper layer - between hidden and output layer
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                //Picking a value between (-initBound, initBound).
                this.w2[i][j] = (random.nextDouble() * (initBound - (-initBound))) + (-initBound);
            }
        }

        //Random values for biases of upper layer - between hidden and output layer
        for (int i = 0; i < this.no; i++) {
            //Picking a value between (-initBound, initBound).
            this.b2[i] = (random.nextDouble() * (initBound - (-initBound))) + (-initBound);
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
            this.z1[i] = sum + this.b1[i];
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
            this.z2[i] = sum + this.b2[i];
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
            //for last layer after further calculation and simplification.
            //Ref: https://www.ics.uci.edu/~pjsadows/notes.pdf
            //NOTE: In actual formula their is a minus (-) sign in front of the delta but while applying the weight
            //changes we will use addition in place of subtraction and the minus sign of learning rate is cancelled by
            //this negative sign.
            delta2[i] = (target[i] - this.o[i]);
        }

        //Weight difference for upper layer - between hidden units and output units
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                this.dw2[i][j] += this.h[i] * delta2[j];
            }
        }
        //Bias difference for upper layer - between hidden units and output units
        System.arraycopy(delta2, 0, this.db2, 0, delta2.length);

        //Derivatives of the activations of the lower layer - between input and hidden layers
        double[] derivatives = this.hiddenActivationFn.squashDerivative(this.z1);

        //Delta for lower layer - between input and hidden layers
        double[] delta1 = new double[this.nh];
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

        //Weight difference for lower layer - between input and hidden layers
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                this.dw1[i][j] += this.input[i] * delta1[j];
            }
        }

        //Bias difference of the activations of the lower layer
        System.arraycopy(delta1, 0, this.db1, 0, delta1.length);
    }

    /**
     * Function to update the weights/biases for the mlp from with the weight/bias changes
     *
     * @param nSamples number of samples on which weight/bias changes are accumulated
     */
    private void updateWeights(int nSamples) {
        //Update weights in lower layer
        for (int i = 0; i < this.ni; i++) {
            for (int j = 0; j < this.nh; j++) {
                //A positive sign is used because while calculating the delta we left out the minus sign there. So that
                //minus sign cancels the minus sign here.
                this.w1[i][j] += this.learningRate * (this.dw1[i][j] / nSamples);
            }
        }

        //Update bias in lower layer
        for (int i = 0; i < this.nh; i++) {
            //A positive sign is used because while calculating the delta we left out the minus sign there. So that
            //minus sign cancels the minus sign here.
            this.b1[i] += this.learningRate * (this.db1[i] / nSamples);
        }

        //Update weights in upper layer
        for (int i = 0; i < this.nh; i++) {
            for (int j = 0; j < this.no; j++) {
                //A positive sign is used because while calculating the delta we left out the minus sign there. So that
                //minus sign cancels the minus sign here.
                this.w2[i][j] += this.learningRate * (this.dw2[i][j] / nSamples);
            }
        }

        //Update bias in upper layer
        for (int i = 0; i < this.no; i++) {
            //A positive sign is used because while calculating the delta we left out the minus sign there. So that
            //minus sign cancels the minus sign here.
            this.b2[i] += this.learningRate * (this.db2[i] / nSamples);
        }

        //Reset the weight changes to zeroes.
        this.dw1 = new double[ni][nh];
        this.dw2 = new double[nh][no];
        this.db1 = new double[nh];
        this.db2 = new double[no];
    }

    /**
     * The method to train the mlp with a particular input and output
     *
     * @param x input
     * @param y output
     */
    public void fit(double x[][], double y[][]) {
        System.out.println("Epoch;Loss");
        for (int epoch = 1; epoch <= this.epochs; epoch++) {
            //Start of an epoch
            double loss = 0;
            //A counter to keep track of batches in mini-batch gradient descent
            int nSamples = 0;
            //For each example
            for (int i = 0; i < x.length; i++) {
                //Do a forward pass
                this.forward(x[i]);
                //Calculate the error
                loss += this.lossFn.calculate(this.o, y[i]);
                //Calculate the weight updates using back-propagation
                this.backward(y[i]);

                nSamples++;
                //Update the weights with the changes
                if (nSamples == this.batchSize) {
                    updateWeights(nSamples);
                    nSamples = 0;
                }
            }
            //When the number of training samples are not exactly divided by the batch size then their will be residual
            //samples in the last batch. Updating changes due to those.
            if (nSamples != 0) {
                updateWeights(nSamples);
            }
            System.out.println(String.format("%s;%s", epoch, loss / x.length));
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
        //The length of predicted output and target output must be same.
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

    /**
     * Print information about this MLP
     *
     * @param showWeights whether to show the weights
     */
    public void printInfo(boolean showWeights) {
        //Print the hyper-parameter
        System.out.println("***********************");
        System.out.println("Hyper parameters");
        System.out.println("***********************");
        System.out.println(String.format("Configuration: %s X %s X %s", this.ni, this.nh, this.no));
        System.out.println("Hidden layer activation function: " + this.hiddenActivationFn.getClass().getName());
        System.out.println("Output layer activation function: " + this.outputActivationFn.getClass().getName());
        System.out.println("Loss function: " + this.lossFn.getClass().getName());
        System.out.println("Epochs: " + this.epochs);
        System.out.println("Learning rate: " + this.learningRate);
        System.out.println("(Gradient Descent) Batch size: " + this.batchSize);
        System.out.println("Random seed: " + this.randomState);

        //If not want to print weights of the MLP
        if (!showWeights) {
            return;
        }

        //Print the weights in lower layer
        System.out.println();
        System.out.println("***********************");
        System.out.println("Weights of lower layer");
        System.out.println("***********************");
        System.out.println();
        for (int i = 0; i < ni; i++) {
            System.out.print("| ");
            for (int j = 0; j < nh; j++) {
                System.out.print(String.format("%4.5f | ", this.w1[i][j]));
            }
            System.out.println();
        }

        //Print the weights in upper layer
        System.out.println();
        System.out.println("***********************");
        System.out.println("Weights of upper layer");
        System.out.println("***********************");
        System.out.println();
        for (int i = 0; i < nh; i++) {
            System.out.print("| ");
            for (int j = 0; j < no; j++) {
                System.out.print(String.format("%4.5f | ", this.w2[i][j]));
            }
            System.out.println();
        }
    }
}
