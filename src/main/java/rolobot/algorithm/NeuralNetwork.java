package rolobot.algorithm;

import static rolobot.Config.ALPHA_NET;

import Jama.Matrix;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.Random;
import robocode.RobocodeFileOutputStream;

/**
 * TODO: Cite textbook for algorithm
 */
public class NeuralNetwork implements ValueFunctionLearner {
    public File neuralNetFile;

    private static final int INPUT_NODES = 4;
    private static final int HIDDEN_NODES = 5;

    private Matrix v = new Matrix(INPUT_NODES, HIDDEN_NODES);
    private Matrix v0 = new Matrix(INPUT_NODES, 1);
    private Matrix w = new Matrix(HIDDEN_NODES, 1);
    private double w0;

    @Override
    public double outputFor(Matrix x) {
        // Feedforward
        Matrix z = activationFunction(x.times(v).plus(v0));
        return activationFunction(z.times(w).get(0, 0) + w0);
    }

    @Override
    public double train(Matrix x, double t) {
        // Feedforward
        Matrix z_in = x.times(v).plus(v0);
        Matrix z = activationFunction(z_in);
        double y_in = z.times(w).get(0, 0) + w0;
        double y = activationFunction(y_in);

        // Backpropagation of error
        double y_error = (t - y) * activationFunctionDerivative(y_in);
        Matrix z_error = w.timesEquals(y_error).arrayTimes(activationFunctionDerivative(z_in));

        double w0_correction = ALPHA_NET * y_error;
        Matrix w_correction = z.timesEquals(w0_correction);

        Matrix v0_correction = z_error.timesEquals(ALPHA_NET);
        Matrix v_correction = v0_correction.times(x);

        // Update weights and biases
        w.plusEquals(w_correction);
        v.plusEquals(v_correction);

        return y_error;
    }

    @Override
    public void load(File file) throws IOException {

    }

    @Override
    public void save(File file) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new RobocodeFileOutputStream(file));
            NumberFormat format = NumberFormat.getNumberInstance(Locale.US);

            v.print(pw, format, 7);
            v0.print(pw, format, 7);
            w.print(pw, format, 7);
            (new Matrix(new double[][]{{w0}})).print(pw, format, 7);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (pw != null) {
                pw.close();
            }
        }
    }

    /**
     * Set every weight to a random double in [-bound, bound].
     */
    public void initializeWeights(double bound) {
        Random random = new Random();
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                v.set(i, j, 2 * bound * random.nextDouble() - bound);
                if (i == 0) {
                    w.set(j, 0, 2 * bound * random.nextDouble() - bound);
                }
            }
            v0.set(i, 0, 2 * bound * random.nextDouble() - bound);
        }
        w0 = 2 * bound * random.nextDouble() - bound;
    }

    /**
     * Bipolar sigmoid.
     */
    private double activationFunction(double input) {
        return (2 / (1 + Math.exp(-input))) - 1;
    }

    private Matrix activationFunction(Matrix input) {
        for (int row = 0; row < input.getRowDimension(); row++) {
            for (int col = 0; col < input.getColumnDimension(); col++) {
                input.set(row, col, activationFunction(input.get(row, col)));
            }
        }
        return input;
    }

    private double activationFunctionDerivative(double input) {
        return activationFunction(input) * (1 - activationFunction(input));
    }

    private Matrix activationFunctionDerivative(Matrix input) {
        for (int row = 0; row < input.getRowDimension(); row++) {
            for (int col = 0; col < input.getColumnDimension(); col++) {
                input.set(row, col, activationFunctionDerivative(input.get(row, col)));
            }
        }
        return input;
    }

}
