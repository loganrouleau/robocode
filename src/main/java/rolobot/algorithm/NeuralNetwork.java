package rolobot.algorithm;

import static rolobot.Config.ALPHA_NET;
import static rolobot.Config.HIDDEN_NODES;
import static rolobot.Config.INPUT_NODES;
import static rolobot.Config.STEP_SIZE;
import static rolobot.Config.QSCALE;
import static rolobot.Config.RESET_NET;
import static rolobot.Config.USE_BIPOLAR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import robocode.RobocodeFileOutputStream;
import rolobot.model.ActionState;

public class NeuralNetwork implements ValueFunctionLearner<ActionState> {
    public File NeuralNetpath;

    private double[][] v;
    private double[] v0;
    private double[] w;
    private double w0;
    private static double delta_y = 0;
    private static double[] delta_z = new double[HIDDEN_NODES];
    private static double[][] delta_v = new double[INPUT_NODES][HIDDEN_NODES];
    private static double[] delta_w = new double[HIDDEN_NODES];
    private static double[] delta_v0 = new double[HIDDEN_NODES];
    private static double delta_w0 = 0;

    public NeuralNetwork(File NeuralNetpath) {
        this.NeuralNetpath = NeuralNetpath;
        try {
            load(NeuralNetpath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public double outputFor(ActionState actionState) {
        int myX = actionState.getX();
        int myY = actionState.getY();
        int enemyX = actionState.getEnemyX();
        int enemyY = actionState.getEnemyY();
        ActionState.Action action = actionState.getAction();

        double[] x = new double[]{(((double) myX) / 2) - 1, ((double) myY) - 1, (((double) enemyX) / 2) - 1, (((double) enemyY) / 1.5) - 1, -1, -1, -1, -1, -1};
        x[8 - action.label()] = 1;
        double y;
        double[] z_in = new double[HIDDEN_NODES];
        double[] z = new double[HIDDEN_NODES];
        double[] y_in = new double[1];
        double[] ones = new double[HIDDEN_NODES];
        Arrays.fill(ones, 1);
        // Feedforward
        z_in = arrayAdd(v0, arrayTimes(x, v));
        z = sigmoid(z_in, USE_BIPOLAR);
        y_in[0] = w0 + arraySum(arrayElementTimes(1, z, w));
        y = QSCALE * sigmoid(y_in, USE_BIPOLAR)[0];
        return y;
    }

    @Override
    public double train(ActionState actionState, double t) {
        int myX = actionState.getX();
        int myY = actionState.getY();
        int enemyX = actionState.getEnemyX();
        int enemyY = actionState.getEnemyY();
        ActionState.Action action = actionState.getAction();

        double[] x = new double[]{(((double) myX) / 2) - 1, ((double) myY) - 1, (((double) enemyX) / 2) - 1, (((double) enemyY) / 1.5) - 1, -1, -1, -1, -1, -1};
        x[8 - action.label()] = 1;
        t = t / QSCALE;
        double y;
        double[] z_in = new double[HIDDEN_NODES];
        double[] z = new double[HIDDEN_NODES];
        double[] y_in = new double[1];
        double[] ones = new double[HIDDEN_NODES];
        Arrays.fill(ones, 1);
        // Feedforward
        z_in = arrayAdd(v0, arrayTimes(x, v));
        z = sigmoid(z_in, USE_BIPOLAR);
        y_in[0] = w0 + arraySum(arrayElementTimes(1, z, w));
        y = sigmoid(y_in, USE_BIPOLAR)[0];
        // 1. Calculate delta for the output neuron
        if (USE_BIPOLAR) {
            delta_y = (t - y) * 0.5 * (1 + y) * (1 - y);
        } else {
            delta_y = (t - y) * y * (1 - y);
        }
        // 3. Calculate delta for the hidden neurons
        if (USE_BIPOLAR) {
            delta_z = arrayElementTimes(0.5 * delta_y, w, arrayAdd(ones, z), arrayAdd(ones, arrayElementTimes(-1, z)));
        } else {
            delta_z = arrayElementTimes(delta_y, w, z, arrayAdd(ones, arrayElementTimes(-1, z)));
        }
        // 2. Update the weights to the output neuron
        delta_w = arrayAdd(arrayElementTimes(STEP_SIZE * delta_y, z), arrayElementTimes(ALPHA_NET, delta_w));
        delta_w0 = STEP_SIZE * delta_y + ALPHA_NET * delta_w0;
        w = arrayAdd(w, delta_w);
        w0 += delta_w0;
        // 4. Update the weights to the hidden neurons
        delta_v = getDeltaV(delta_v, delta_z, x, STEP_SIZE, ALPHA_NET);
        delta_v0 = arrayAdd(arrayElementTimes(STEP_SIZE, delta_z), arrayElementTimes(ALPHA_NET, delta_v0));
        v = arrayAdd2D(v, delta_v);
        v0 = arrayAdd(v0, delta_v0);
        return 0;
    }

    @Override
    public void load(File file) throws IOException {
        double v[][] = new double[INPUT_NODES][HIDDEN_NODES];
        if (RESET_NET) {
            v = randomizeWeights2DArray(INPUT_NODES, HIDDEN_NODES);
        } else {
            try {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(new FileReader(file));
                    String sCurrentLine;
                    int i = 0;
                    while ((sCurrentLine = reader.readLine()) != null) {
                        String[] arr = sCurrentLine.split(" ");
                        if (i < INPUT_NODES) {
                            for (int j = 0; j < HIDDEN_NODES; j++) {
                                v[i][j] = Double.parseDouble(arr[j]);
                            }
                        }
                        i++;
                    }
                } finally {
                    if (reader != null) {
                        reader.close();
                    }
                }
            } catch (IOException e) {
            }
        }
        this.v = v;

        double v0[] = new double[HIDDEN_NODES];
        if (RESET_NET) {
            v0 = randomizeWeights1DArray(HIDDEN_NODES);


        } else {
            try {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(new FileReader(file));
                    String sCurrentLine;
                    int i = 0;
                    while ((sCurrentLine = reader.readLine()) != null) {
                        String[] arr = sCurrentLine.split(" ");
                        if (i == INPUT_NODES) {
                            for (int j = 0; j < HIDDEN_NODES; j++) {
                                v0[j] = Double.parseDouble(arr[j]);
                            }
                        }
                        i++;
                    }
                } finally {
                    if (reader != null) {
                        reader.close();
                    }
                }
            } catch (IOException e) {
            }
        }
        this.v0 = v0;

        double w[] = new double[HIDDEN_NODES];
        if (RESET_NET) {
            w = randomizeWeights1DArray(HIDDEN_NODES);
        } else {
            try {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(new FileReader(file));
                    String sCurrentLine;
                    int i = 0;
                    while ((sCurrentLine = reader.readLine()) != null) {
                        String[] arr = sCurrentLine.split(" ");
                        if (i == INPUT_NODES + 1) {
                            for (int j = 0; j < HIDDEN_NODES; j++) {
                                w[j] = Double.parseDouble(arr[j]);
                            }
                        }
                        i++;
                    }
                } finally {
                    if (reader != null) {
                        reader.close();
                    }
                }
            } catch (IOException e) {
            }
        }
        this.w = w;


        double w0 = 0;
        if (RESET_NET) {
            w0 = 0.5 * Math.random() - 0.25;
//w0 = 0;
        } else {
            try {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(new FileReader(file));
                    String sCurrentLine;
                    int i = 0;
                    while ((sCurrentLine = reader.readLine()) != null) {
                        String[] arr = sCurrentLine.split(" ");
                        if (i == INPUT_NODES + 2) {
                            w0 = Double.parseDouble(arr[0]);
                        }
                        i++;
                    }
                } finally {
                    if (reader != null) {
                        reader.close();
                    }
                }
            } catch (IOException e) {
            }
        }
        this.w0 = w0;
    }

    @Override
    public void save(File file) {
        PrintStream s = null;
        try {
            s = new PrintStream(new RobocodeFileOutputStream(file));
            for (int i = 0; i < INPUT_NODES; i++) {
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    s.printf("%.4f ", v[i][j]);
                }
                s.printf("%n");
            }
            for (int i = 0; i < HIDDEN_NODES; i++) {
                s.printf("%.4f ", v0[i]);
            }
            s.printf("%n");
            for (int i = 0; i < HIDDEN_NODES; i++) {
                s.printf("%.4f ", w[i]);
            }
            s.printf("%n%.4f", w0);
        } catch (IOException e) {
        } finally {
            if (s != null) {
                s.close();
            }
        }
    }

    private static double[] sigmoid(double[] input_array, boolean use_bipolar) {
        double[] output_array = new double[input_array.length];
        for (int i = 0; i < input_array.length; i++) {
            if (use_bipolar) {
                output_array[i] = (2 / (1 + Math.exp(-input_array[i]))) - 1;
            } else {
                output_array[i] = 1 / (1 + Math.exp(-input_array[i]));


            }
        }
        return output_array;
    }

    private static double[][] getDeltaV(double[][] delta_v_prev, double[] delta_z, double[] x, double p, double alpha) {
        double delta_v[][] = new double[delta_v_prev.length][delta_v_prev[0].length];
        for (int i = 0; i < delta_v.length; i++) {
            for (int j = 0; j < delta_v[0].length; j++) {
                delta_v[i][j] += p * delta_z[j] * x[i] + alpha * delta_v_prev[i][j];
            }
        }
        return delta_v;
    }

    private static double[] randomizeWeights1DArray(int hidden_nodes) {
        double[] result = new double[hidden_nodes];
        for (int i = 0; i < hidden_nodes; i++) {
            result[i] = 0.5 * Math.random() - 0.25;
        }
        return result;
    }

    private static double[][] randomizeWeights2DArray(int input_nodes, int hidden_nodes) {
        double[][] result = new double[input_nodes][hidden_nodes];
        for (int i = 0; i < input_nodes; i++) {
            for (int j = 0; j < hidden_nodes; j++) {
                result[i][j] = 0.5 * Math.random() - 0.25;
            }
        }
        return result;
    }

    private static double[] arrayTimes(double[] first_array, double[][] second_array) {
        double[] output_array = new double[second_array[0].length];
        double sum;
        for (int i = 0; i < output_array.length; i++) {
            sum = 0;
            for (int j = 0; j < first_array.length; j++) {
                sum += first_array[j] * second_array[j][i];
            }
            output_array[i] = sum;
        }
        return output_array;
    }

    private static double[] arrayElementTimes(double scalar, double[]... input_args) {
        double sum;
        double[] result = new double[input_args[0].length];
        for (int i = 0; i < input_args[0].length; i++) {
            sum = 1;
            for (int j = 0; j < input_args.length; j++) {
                sum *= input_args[j][i];
            }
            result[i] += scalar * sum;
        }
        return result;
    }

    private static double arraySum(double... values) {
        double result = 0;
        for (double value : values)
            result += value;
        return result;
    }

    private static double[] arrayAdd(double[] array1, double[] array2) {
        double[] result = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            result[i] = array1[i] + array2[i];
        }
        return result;
    }

    private static double[][] arrayAdd2D(double[][] array1, double[][] array2) {
        for (int i = 0; i < array1.length; i++) {
            for (int j = 0; j < array1[0].length; j++) {
                array1[i][j] += array2[i][j];
            }
        }
        return array1;
    }

}
