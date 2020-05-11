import mlp.MultilayerPerceptron;
import mlp.loss_functions.Loss;
import org.junit.Test;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 10-05-2020 23:46
 * Purpose: TODO:
 **/
public class TestMLP {

    @Test
    public void testForward() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
        int ni = 2;
        int nh = 2;
        int no = 2;
        double[] input = {0.5, 0.1};
        double[] target = {0.1, 0.99};
        double[][] w1 = {{0.15, 0.2}, {0.25, 0.3}};
        double[][] w2 = {{0.4, 0.45}, {0.5, 0.55}};

        double[] expectedH = {0.525, 0.532};
        double[] expectedO = {0.617, 0.629};
        double expectedLoss = 0.199;

        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20);
        Field w1Field = mlp.getClass().getDeclaredField("w1");
        w1Field.setAccessible(true);
        w1Field.set(mlp, w1);
        Field w2Field = mlp.getClass().getDeclaredField("w2");
        w2Field.setAccessible(true);
        w2Field.set(mlp, w2);

        Method forwardMethod = mlp.getClass().getDeclaredMethod("forward", double[].class);
        forwardMethod.setAccessible(true);
        forwardMethod.invoke(mlp, new Object[]{input});

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
        //checking that activations at hidden layer and output layer are expected.
        assert Arrays.equals(expectedH, h);
        assert Arrays.equals(expectedO, o);

        Field lossFunctionField = mlp.getClass().getDeclaredField("lossFunction");
        lossFunctionField.setAccessible(true);
        Loss lossFunction = (Loss) lossFunctionField.get(mlp);
        double loss = lossFunction.calculate(target, (double[]) oField.get(mlp));
        //Rounding to 3 decimal places
        loss = Math.round(loss * 1000) / 1000.0;
        //checking we are getting expected loss. It is expected to be a mean squared loss.
        assert loss == expectedLoss;
    }

    @Test
    public void testBackward() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
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

        double[][] expectedDw2 = {{-0.082, 0.023}, {-0.083, 0.023}};
        double[][] expectedDw1 = {{-0.000439, -0.000498}, {-0.000877, -0.000995}};

        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, learningRate);
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

        Method backwardMethod = mlp.getClass().getDeclaredMethod("backward", double[].class);
        backwardMethod.setAccessible(true);
        backwardMethod.invoke(mlp, new Object[]{target});

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
        //checking that activations at hidden layer and output layer are expected.
        assert Arrays.deepEquals(expectedDw2, dw2);
        assert Arrays.deepEquals(expectedDw1, dw1);
    }

    @Test
    public void testUpdateWeights() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
        int ni = 2;
        int nh = 2;
        int no = 2;
        double[][] w1 = {{0.15, 0.25}, {0.2, 0.3}};
        double[][] w2 = {{0.4, 0.5}, {0.45, 0.55}};
        double learningRate = 0.5;
        double[][] dw2 = {{-0.082, 0.023}, {-0.083, 0.023}};
        double[][] dw1 = {{-0.000439, -0.000498}, {-0.000877, -0.000995}};

        double[][] expectedW1 = {{0.14978, 0.249751}, {0.199562, 0.299503}};
        double[][] expectedW2 = {{0.359, 0.511}, {0.409, 0.562}};

        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, learningRate);
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

        Method backwardMethod = mlp.getClass().getDeclaredMethod("updateWeights");
        backwardMethod.setAccessible(true);
        backwardMethod.invoke(mlp);
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

        //checking that activations at hidden layer and output layer are expected.
        assert Arrays.deepEquals(expectedW2, w2);
        assert Arrays.deepEquals(expectedW1, w1);
    }
}
