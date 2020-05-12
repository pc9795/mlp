import mlp.activations.*;
import org.junit.Test;

import java.util.Arrays;

/**
 * Created By: Prashant Chaubey
 * Created On: 11-05-2020 13:42
 * Purpose: Test for all the implementations of the `mlp.activations.ActivationFn`
 **/
public class TestActivationFn {

    @Test
    public void testLinearActivationFn() {
        ActivationFn activationFn = new LinearActivationFn();

        assert Arrays.equals(activationFn.squash(new double[]{0.5}), new double[]{0.5});
        assert Arrays.equals(activationFn.squashDerivative(new double[]{0.5}), new double[]{1});
    }

    @Test
    public void testReluActivationFn() {
        ActivationFn activationFn = new ReluActivationFn();

        assert Arrays.equals(activationFn.squash(new double[]{-0.5, 0.5}), new double[]{0, 0.5});
        assert Arrays.equals(activationFn.squashDerivative(new double[]{-0.5, 0.5}), new double[]{0, 1});
    }

    @Test
    public void testSigmoidActivationFn() {
        ActivationFn activationFn = new SigmoidActivationFn();

        assert Arrays.equals(roundTo4Places(activationFn.squash(new double[]{0.5})), new double[]{0.6225});
        assert Arrays.equals(roundTo4Places(activationFn.squashDerivative(new double[]{0.5})), new double[]{0.2350});
    }

    @Test
    public void testTanhActivationFn() {
        ActivationFn activationFn = new TanhActivationFn();

        assert Arrays.equals(roundTo4Places(activationFn.squash(new double[]{0.5})), new double[]{0.4621});
        assert Arrays.equals(roundTo4Places(activationFn.squashDerivative(new double[]{0.5})), new double[]{0.7864});
    }

    @Test
    public void testLeakReluActivationFn() {
        ActivationFn activationFn = new LeakyReluActivationFn();

        assert Arrays.equals(activationFn.squash(new double[]{-0.5, 0.5}), new double[]{-0.005, 0.5});
        assert Arrays.equals(activationFn.squashDerivative(new double[]{-0.5, 0.5}), new double[]{0.01, 1});
    }

    @Test
    public void testSoftmaxActivationFn() {
        ActivationFn activationFn = new SoftmaxActivationFn();

        assert Arrays.equals(roundTo4Places(activationFn.squash(new double[]{3.0, 1.0, 0.2})), new double[]{0.836, 0.1131,
                0.0508});
    }

    private double[] roundTo4Places(double[] x) {
        double[] output = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            output[i] = Math.round(x[i] * Math.pow(10, 4)) / Math.pow(10, 4);
        }
        return output;
    }
}
