import mlp.loss_functions.BinaryCrossEntropyLossFn;
import mlp.loss_functions.CategoricalCrossEntropyLossFn;
import mlp.loss_functions.LossFn;
import mlp.loss_functions.SquaredErrorLossFn;
import org.junit.Test;

/**
 * Created By: Prashant Chaubey
 * Created On: 13-05-2020 02:17
 * Purpose: Test-cases for implementations of interface `LossFn`
 **/
public class TestLossFn {

    @Test
    public void testSquaredErrorLossFn() {
        LossFn lossFn = new SquaredErrorLossFn();

        assert lossFn.calculate(new double[]{3, 0.5, 2}, new double[]{4, 1, 2}) == 0.625;
    }

    @Test
    public void testBinaryCrossEntropyLossFn() {
        LossFn lossFn = new BinaryCrossEntropyLossFn();
        //In case of binary cross entropy loss the output activation function will be sigmoid which always returns
        //values between 0 and 1.
        assert lossFn.calculate(new double[]{0.8, 0.2, 0.8}, new double[]{1, 0, 1}) == 0.6694306539426291;
    }

    @Test
    public void testCategoricalCrossEntropyLossFn() {
        LossFn lossFn = new CategoricalCrossEntropyLossFn();
        //In case of categorical cross entropy loss the predicted values sum to 1 and in the true values only one value
        //will be 1.
        assert lossFn.calculate(new double[]{0.05, 0.2, 0.6, 0.1, 0.05}, new double[]{0, 0, 1, 0, 0}) == 0.5108256237659907;
    }
}
