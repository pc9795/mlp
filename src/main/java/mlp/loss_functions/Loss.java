package mlp.loss_functions;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:28
 * Purpose: TODO:
 **/
public interface Loss {
    double calculate(double[] output, double[] target);
}
