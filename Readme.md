Import this project as a maven project in your preferred IDE
* **Intellij** - https://www.lagomframework.com/documentation/1.6.x/java/IntellijMaven.html
* **Eclipse** - https://vaadin.com/learn/tutorials/import-maven-project-eclipse

### Project Structure

* `experiments` - All the code for the experiments ran. XOR, Sin and Letter recognition
* `experiments.utils` - Utility methods which are used in evaluating the experiments
* `mlp` - All the code for Multi layer perceptron implementation
* `mlp.activations` - All the activation functions which can be used - RELU, Leaky RELU, Sigmoid, Linear, Tanh, Softmax
* `mlp.exceptions` - Custom exceptions for this project
* `mlp.loss_functions` - All the loss function which can be used - Squared loss, Cross entropy, Binary cross entropy

Sample Training and testing Example
```
    int ni = ...
    int nh = ...
    int no = ...
    int randomState = ...
    double learningRate = ...
    int epochs = ...
    ActivationType type = ...
    boolean isClassification = ...
    boolean isMulticlass = ...
    int batchSize = ...
    
    //Create an multi layer perceptron object
    MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, randomState, learningRate, epochs, type, 
        isClassification, isMulticlass, bathcSize);

    //Training the MLP
    mlp.fit(input, output);

    //Get the predictions of the MLP
    double predicted[][] = mlp.predict(input);
```