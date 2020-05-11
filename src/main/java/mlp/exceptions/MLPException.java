package mlp.exceptions;

/**
 * Created By: Prashant Chaubey
 * Created On: 09-05-2020 02:35
 * Purpose: Exception if the library is not used in the expected manner
 **/
public class MLPException extends RuntimeException {
    public MLPException(String message) {
        super(message);
    }
}
