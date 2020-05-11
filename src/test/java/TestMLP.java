import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;
import mlp.loss_functions.LossFn;
import mlp.loss_functions.SquaredErrorLossFn;
import org.junit.Test;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 10-05-2020 23:46
 * Purpose: Tests for class `mlp.MultilayerPerceptron`
 **/
public class TestMLP {

    @Test
    public void testForward() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
        //Initialization parameters
        int ni = 2;
        int nh = 2;
        int no = 2;
        double[] input = {0.5, 0.1};
        double[] target = {0.1, 0.99};
        double[][] w1 = {{0.15, 0.2}, {0.25, 0.3}};
        double[][] w2 = {{0.4, 0.45}, {0.5, 0.55}};
        //Expected parameters
        double[] expectedH = {0.525, 0.532};
        double[] expectedO = {0.617, 0.629};
        double expectedLoss = 0.199;
        //Setting the initial values
        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, 0.1,
                500, ActivationType.SIGMOID, true, false);
        Field w1Field = mlp.getClass().getDeclaredField("w1");
        w1Field.setAccessible(true);
        w1Field.set(mlp, w1);
        Field w2Field = mlp.getClass().getDeclaredField("w2");
        w2Field.setAccessible(true);
        w2Field.set(mlp, w2);
        Field lossFnFiled = mlp.getClass().getDeclaredField("lossFn");
        lossFnFiled.setAccessible(true);
        lossFnFiled.set(mlp, new SquaredErrorLossFn());
        //Calling the method
        Method forwardMethod = mlp.getClass().getDeclaredMethod("forward", double[].class);
        forwardMethod.setAccessible(true);
        forwardMethod.invoke(mlp, new Object[]{input});
        //Checking the expected values
        Field hField = mlp.getClass().getDeclaredField("h");
        hField.setAccessible(true);
        double[] h = (double[]) hField.get(mlp);
        Field oField = mlp.getClass().getDeclaredField("o");
        oField.setAccessible(true);
        double[] o = (double[]) oField.get(mlp);
        //Rounding to 3 decimal places.
        for (int i = 0; i < nh; i++) {
            h[i] = Math.round(h[i] * 1000) / 1000.0;
        }
        for (int i = 0; i < no; i++) {
            o[i] = Math.round(o[i] * 1000) / 1000.0;
        }
        //Checking that activations at hidden layer and output layer are expected.
        assert Arrays.equals(expectedH, h);
        assert Arrays.equals(expectedO, o);

        Field lossFunctionField = mlp.getClass().getDeclaredField("lossFn");
        lossFunctionField.setAccessible(true);
        LossFn lossFnFunction = (LossFn) lossFunctionField.get(mlp);
        double loss = lossFnFunction.calculate(target, (double[]) oField.get(mlp));
        //Rounding to 3 decimal places
        loss = Math.round(loss * 1000) / 1000.0;
        //Checking we are getting expected loss. It is expected to be a mean squared loss.
        assert loss == expectedLoss;
    }

    @Test
    public void testBackward() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
        //Initialization parameters
        int ni = 2;
        int nh = 2;
        int no = 2;
        double[] input = {0.05, 0.1};
        double[] target = {0.01, 0.99};
        double[] h = {0.593269992, 0.596884378};
        double[] z1 = {0.3775, 0.3925};
        double[] z2 = {1.105905967, 1.2249};
        double[] o = {0.75136507, 0.772928465};
        double[][] w1 = {{0.15, 0.25}, {0.2, 0.3}};
        double[][] w2 = {{0.4, 0.5}, {0.45, 0.55}};
        double learningRate = 0.5;
        int epochs = 1000;
        //Expected parameters
        double[][] expectedDw2 = {{-0.082, 0.023}, {-0.083, 0.023}};
        double[][] expectedDw1 = {{-0.000439, -0.000498}, {-0.000877, -0.000995}};
        //Setting the initial values
        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, learningRate, epochs,
                ActivationType.SIGMOID, true, false);
        Field w1Field = mlp.getClass().getDeclaredField("w1");
        w1Field.setAccessible(true);
        w1Field.set(mlp, w1);
        Field w2Field = mlp.getClass().getDeclaredField("w2");
        w2Field.setAccessible(true);
        w2Field.set(mlp, w2);
        Field hField = mlp.getClass().getDeclaredField("h");
        hField.setAccessible(true);
        hField.set(mlp, h);
        Field oField = mlp.getClass().getDeclaredField("o");
        oField.setAccessible(true);
        oField.set(mlp, o);
        Field inputFiled = mlp.getClass().getDeclaredField("input");
        inputFiled.setAccessible(true);
        inputFiled.set(mlp, input);
        Field z1Field = mlp.getClass().getDeclaredField("z1");
        z1Field.setAccessible(true);
        z1Field.set(mlp, z1);
        Field z2Field = mlp.getClass().getDeclaredField("z2");
        z2Field.setAccessible(true);
        z2Field.set(mlp, z2);
        Field lossFnFiled = mlp.getClass().getDeclaredField("lossFn");
        lossFnFiled.setAccessible(true);
        lossFnFiled.set(mlp, new SquaredErrorLossFn());
        //Calling the method
        Method backwardMethod = mlp.getClass().getDeclaredMethod("backward", double[].class);
        backwardMethod.setAccessible(true);
        backwardMethod.invoke(mlp, new Object[]{target});
        //Checking the expected values
        Field dw1Field = mlp.getClass().getDeclaredField("dw1");
        dw1Field.setAccessible(true);
        double[][] dw1 = (double[][]) dw1Field.get(mlp);
        Field dw2Field = mlp.getClass().getDeclaredField("dw2");
        dw2Field.setAccessible(true);
        double[][] dw2 = (double[][]) dw2Field.get(mlp);
        //Rounding to 6 decimal places. These weights changes will be much smaller.
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nh; j++) {
                dw1[i][j] = Math.round(dw1[i][j] * 1000_000) / 1000_000.0;
            }
        }
        //Rounding to 3 decimal places.
        for (int i = 0; i < nh; i++) {
            for (int j = 0; j < no; j++) {
                dw2[i][j] = Math.round(dw2[i][j] * 1000) / 1000.0;
            }
        }
        //Checking that activations at hidden layer and output layer are expected.
        assert Arrays.deepEquals(expectedDw2, dw2);
        assert Arrays.deepEquals(expectedDw1, dw1);
    }

    @Test
    public void testUpdateWeights() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
        //Initialization parameters
        int ni = 2;
        int nh = 2;
        int no = 2;
        double[][] w1 = {{0.15, 0.25}, {0.2, 0.3}};
        double[][] w2 = {{0.4, 0.5}, {0.45, 0.55}};
        double learningRate = 0.5;
        double[][] dw2 = {{-0.082, 0.023}, {-0.083, 0.023}};
        double[][] dw1 = {{-0.000439, -0.000498}, {-0.000877, -0.000995}};
        int epochs = 1000;
        //Expected parameters
        double[][] expectedW1 = {{0.14978, 0.249751}, {0.199562, 0.299503}};
        double[][] expectedW2 = {{0.359, 0.511}, {0.409, 0.562}};
        //Setting the initial values
        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, learningRate, epochs,
                ActivationType.SIGMOID, true, false);
        Field w1Field = mlp.getClass().getDeclaredField("w1");
        w1Field.setAccessible(true);
        w1Field.set(mlp, w1);
        Field w2Field = mlp.getClass().getDeclaredField("w2");
        w2Field.setAccessible(true);
        w2Field.set(mlp, w2);
        Field dw1Field = mlp.getClass().getDeclaredField("dw1");
        dw1Field.setAccessible(true);
        dw1Field.set(mlp, dw1);
        Field dw2Field = mlp.getClass().getDeclaredField("dw2");
        dw2Field.setAccessible(true);
        dw2Field.set(mlp, dw2);
        Field lossFnFiled = mlp.getClass().getDeclaredField("lossFn");
        lossFnFiled.setAccessible(true);
        lossFnFiled.set(mlp, new SquaredErrorLossFn());
        //Calling the method
        Method backwardMethod = mlp.getClass().getDeclaredMethod("updateWeights");
        backwardMethod.setAccessible(true);
        backwardMethod.invoke(mlp);
        //Checking the expected values
        w1 = (double[][]) w1Field.get(mlp);
        w2 = (double[][]) w2Field.get(mlp);
        //Rounding to 6 decimal places. These weights changes will be much smaller.
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nh; j++) {
                w1[i][j] = Math.round(w1[i][j] * 1000_000) / 1000_000.0;
            }
        }
        //Rounding to 3 decimal places.
        for (int i = 0; i < nh; i++) {
            for (int j = 0; j < no; j++) {
                w2[i][j] = Math.round(w2[i][j] * 1000) / 1000.0;
            }
        }
        //Checking that activations at hidden layer and output layer are expected.
        assert Arrays.deepEquals(expectedW2, w2);
        assert Arrays.deepEquals(expectedW1, w1);
    }
}
