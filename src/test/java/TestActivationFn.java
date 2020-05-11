import mlp.activations.*;
import org.junit.Test;

/**
 * Created By: Prashant Chaubey
 * Created On: 11-05-2020 13:42
 * Purpose: Test for all the implementations of the `mlp.activations.ActivationFn`
 **/
public class TestActivationFn {

    @Test
    public void testLinearActivationFn() {
        ActivationFn activationFn = new LinearActivationFn();

        assert activationFn.squash(0.5) == 0.5;
        assert activationFn.squashDerivative(0.5) == 1;
    }

    @Test
    public void testReluActivationFn() {
        ActivationFn activationFn = new ReluActivationFn();

        assert activationFn.squash(-0.5) == 0;
        assert activationFn.squashDerivative(-0.5) == 0;

        assert activationFn.squash(0.5) == 0.5;
        assert activationFn.squashDerivative(0.5) == 1;
    }

    @Test
    public void testSigmoidActivationFn() {
        ActivationFn activationFn = new SigmoidActivationFn();

        assert roundTo4Places(activationFn.squash(0.5)) == 0.6225;
        assert roundTo4Places(activationFn.squashDerivative(0.5)) == 0.2350;
    }

    @Test
    public void testTanhActivationFn() {
        ActivationFn activationFn = new TanhActivationFn();

        assert roundTo4Places(activationFn.squash(0.5)) == 0.4621;
        assert roundTo4Places(activationFn.squashDerivative(0.5)) == 0.7864;
    }

    private double roundTo4Places(double x) {
        return Math.round(x * Math.pow(10, 4)) / Math.pow(10, 4);
    }
}
