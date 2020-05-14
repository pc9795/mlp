import mlp.MultilayerPerceptron;
import mlp.activations.ActivationType;
import mlp.activations.SigmoidActivationFn;
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
        double[] input = {0.05, 0.1};
        double[] target = {0.01, 0.99};
        double[][] w1 = {{0.15, 0.25}, {0.2, 0.3}};
        double[][] w2 = {{0.4, 0.5}, {0.45, 0.55}};
        double[] b1 = {0.35, 0.35};
        double[] b2 = {0.60, 0.60};

        //Expected parameters
        double[] expectedH = {0.593269992, 0.596884378};
        double[] expectedO = {0.75136507, 0.772928465};
        double expectedLoss = 0.298371109;

        //Setting the initial values
        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, 0.1,
                500, ActivationType.SIGMOID, true, false);
        initializedFields(mlp, new String[]{"w1", "w2", "b1", "b2"}, new Object[]{w1, w2, b1, b2});

        //At the time of writing test cases we used squared loss for binary/multi-label classification for getting the
        //values for the tests. This is changed and we are now using binary-cross entropy in those scenarios. Until
        //the calculations of these test cases are updated, we are manually injecting the squared error loss as a loss
        //function.
        initializedFields(mlp, new String[]{"lossFn"}, new Object[]{new SquaredErrorLossFn()});

        //Calling the method
        Method forwardMethod = mlp.getClass().getDeclaredMethod("forward", double[].class);
        forwardMethod.setAccessible(true);
        forwardMethod.invoke(mlp, new Object[]{input});

        //Getting the fields that need to be checked
        Field hField = mlp.getClass().getDeclaredField("h");
        hField.setAccessible(true);
        double[] h = (double[]) hField.get(mlp);
        Field oField = mlp.getClass().getDeclaredField("o");
        oField.setAccessible(true);
        double[] o = (double[]) oField.get(mlp);

        //Checking that activations at hidden layer and output layer are expected.
        assert Arrays.equals(expectedH, round(h, 9));
        assert Arrays.equals(expectedO, round(o, 9));

        Field lossFunctionField = mlp.getClass().getDeclaredField("lossFn");
        lossFunctionField.setAccessible(true);
        LossFn lossFnFunction = (LossFn) lossFunctionField.get(mlp);
        double loss = lossFnFunction.calculate((double[]) oField.get(mlp), target);

        //Rounding to 9 decimal places
        loss = Math.round(loss * 1000_000_000) / 1000_000_000.0;

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
        double[] expectedDb1 = {-0.008771, -0.009954};
        double[] expectedDb2 = {-0.138499, 0.038099};

        //This adjustment is only done as at the time of writing these test cases we were using squared error for
        //classification. DON'T TRY TO UNDERSTAND THIS. It will be removed once the new calculations are updated in the
        //tests.
        double outputDerivatives[] = new SigmoidActivationFn().squashDerivative(z2);
        for (int i = 0; i < o.length; i++) {
            target[i] = target[i] * outputDerivatives[i];
            o[i] = o[i] * outputDerivatives[i];
        }

        //Setting the initial values
        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, learningRate, epochs,
                ActivationType.SIGMOID, true, false);
        initializedFields(mlp, new String[]{"w1", "w2", "h", "o", "input", "z1", "z2"}, new Object[]{w1, w2, h, o, input, z1, z2});

        //At the time of writing test cases we used squared loss for binary/mulit-label classification for getting the
        //values for the tests. This is changed and we are now using binary-cross entropy in those scenarios. Until
        //the calculations of these test cases are updated manually injecting the squared error loss as a loss function.
        initializedFields(mlp, new String[]{"lossFn"}, new Object[]{new SquaredErrorLossFn()});

        //Calling the method
        Method backwardMethod = mlp.getClass().getDeclaredMethod("backward", double[].class);
        backwardMethod.setAccessible(true);
        backwardMethod.invoke(mlp, new Object[]{target});

        //Getting the fields that need to be checked
        Field dw1Field = mlp.getClass().getDeclaredField("dw1");
        dw1Field.setAccessible(true);
        double[][] dw1 = (double[][]) dw1Field.get(mlp);
        Field dw2Field = mlp.getClass().getDeclaredField("dw2");
        dw2Field.setAccessible(true);
        double[][] dw2 = (double[][]) dw2Field.get(mlp);
        Field db1Field = mlp.getClass().getDeclaredField("db1");
        db1Field.setAccessible(true);
        double db1[] = (double[]) db1Field.get(mlp);
        Field db2Field = mlp.getClass().getDeclaredField("db2");
        db2Field.setAccessible(true);
        double db2[] = (double[]) db2Field.get(mlp);

        //Checking that activations at hidden layer and output layer are expected.
        assert Arrays.deepEquals(expectedDw2, round(dw2, 3));
        assert Arrays.deepEquals(expectedDw1, round(dw1, 6));
        assert Arrays.equals(expectedDb1, round(db1, 6));
        assert Arrays.equals(expectedDb2, round(db2, 6));
    }

    @Test
    public void testUpdateWeights() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException,
            InvocationTargetException {
        //Initialization parameters
        int ni = 2;
        int nh = 2;
        int no = 2;
        int epochs = 1000;
        double learningRate = 0.5;
        double[][] w1 = {{0.15, 0.25}, {0.2, 0.3}};
        double[][] w2 = {{0.4, 0.5}, {0.45, 0.55}};
        double[][] dw2 = {{-0.082, 0.023}, {-0.083, 0.023}};
        double[][] dw1 = {{-0.000439, -0.000498}, {-0.000877, -0.000995}};
        double[] b1 = {0.35, 0.35};
        double[] b2 = {0.60, 0.60};
        double[] db1 = {-0.008771, -0.009954};
        double[] db2 = {-0.138499, 0.038099};

        //Expected parameters
        double[][] expectedW1 = {{0.14978, 0.249751}, {0.199562, 0.299503}};
        double[][] expectedW2 = {{0.359, 0.511}, {0.409, 0.562}};
        double[] expectedB1 = {0.345615, 0.345023};
        double[] expectedB2 = {0.530751, 0.61905};

        //Setting the initial values
        MultilayerPerceptron mlp = new MultilayerPerceptron(ni, nh, no, 20, learningRate, epochs,
                ActivationType.SIGMOID, true, false);
        initializedFields(mlp, new String[]{"w1", "w2", "dw1", "dw2", "b1", "b2", "db1", "db2"}, new Object[]{w1, w2, dw1, dw2, b1, b2, db1, db2});

        //At the time of writing test cases we used squared loss for binary/mulit-label classification for getting the
        //values for the tests. This is changed and we are now using binary-cross entropy in those scenarios. Until
        //the calculations of these test cases are updated manually injecting the squared error loss as a loss function.
        Field lossFnFiled = mlp.getClass().getDeclaredField("lossFn");
        lossFnFiled.setAccessible(true);
        lossFnFiled.set(mlp, new SquaredErrorLossFn());

        //Calling the method
        Method backwardMethod = mlp.getClass().getDeclaredMethod("updateWeights", int.class);
        backwardMethod.setAccessible(true);
        backwardMethod.invoke(mlp, 1);

        //Getting the fields that need to be checked
        Field w1Field = mlp.getClass().getDeclaredField("w1");
        w1Field.setAccessible(true);
        w1 = (double[][]) w1Field.get(mlp);
        Field w2Field = mlp.getClass().getDeclaredField("w2");
        w2Field.setAccessible(true);
        w2 = (double[][]) w2Field.get(mlp);
        Field b1Field = mlp.getClass().getDeclaredField("b1");
        b1Field.setAccessible(true);
        b1 = (double[]) b1Field.get(mlp);
        Field b2Field = mlp.getClass().getDeclaredField("b2");
        b2Field.setAccessible(true);
        b2 = (double[]) b2Field.get(mlp);

        //Checking that activations at hidden layer and output layer are expected.
        assert Arrays.deepEquals(expectedW2, round(w2, 3));
        assert Arrays.deepEquals(expectedW1, round(w1, 6));
        assert Arrays.equals(expectedB1, round(b1, 6));
        assert Arrays.equals(expectedB2, round(b2, 6));
    }

    /**
     * Round values of a 2d array
     *
     * @param arr    input 2d array
     * @param places places to round each value in the array
     * @return rounded 2d array
     */
    private double[][] round(double[][] arr, int places) {
        double[][] output = new double[arr.length][arr[0].length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                output[i][j] = Math.round(arr[i][j] * Math.pow(10, places)) / Math.pow(10, places);
            }
        }
        return output;
    }

    /**
     * Round values of an array
     *
     * @param arr    input array
     * @param places places to round each value in the array
     * @return rounded array
     */
    private double[] round(double[] arr, int places) {
        double[] output = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            output[i] = Math.round(arr[i] * Math.pow(10, places)) / Math.pow(10, places);
        }
        return output;
    }

    /**
     * Set fields using reflection
     *
     * @param obj    obj whose fields are going to be set
     * @param fields name of the fields
     * @param values values of the fields
     * @throws NoSuchFieldException   if no field with the name
     * @throws IllegalAccessException if not allowed to set values using reflection
     */
    private void initializedFields(Object obj, String[] fields, Object[] values) throws NoSuchFieldException,
            IllegalAccessException {
        assert fields.length == values.length;

        for (int i = 0; i < fields.length; i++) {
            Field field = obj.getClass().getDeclaredField(fields[i]);
            field.setAccessible(true);
            field.set(obj, values[i]);
        }
    }
}
