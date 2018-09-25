import Jama.Matrix;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import rolobot.algorithm.NeuralNetwork;

/**
 * Train a neural net to the exclusive or (XOR) table using 2 input nodes, 4 hidden nodes, and 1 output node. This
 * problem requires a hidden layer because XOR is not a linearly separable function.
 */
public class NeuralNetworkTest {
    private Map<Matrix, Double> trainingVectors = new HashMap<>();
    private NeuralNetwork neuralNet = new NeuralNetwork();

    public static void main(String[] args) {
        new NeuralNetworkTest().run();
    }

    private void run() {
        // The XOR table
        trainingVectors.put(new Matrix(new double[][]{{0, 0}}), 0.0);
        trainingVectors.put(new Matrix(new double[][]{{0, 1}}), 1.0);
        trainingVectors.put(new Matrix(new double[][]{{1, 0}}), 1.0);
        trainingVectors.put(new Matrix(new double[][]{{1, 1}}), 0.0);

        neuralNet.initializeWeights(0.5);
        neuralNet.neuralNetFile = new File("src.main.test.test-net");

        double error = Double.MAX_VALUE;
        int epochs = 0;

        while (Math.abs(error) > 0.001) {
            for (Matrix trainingVector : trainingVectors.keySet()) {
                error = neuralNet.train(trainingVector, trainingVectors.get(trainingVector));
                System.out.println(error);
            }
            epochs++;
        }

        //save net
        System.out.println("\nEpochs: " + epochs);
        for (Matrix trainingVector : trainingVectors.keySet()) {
            System.out.println(neuralNet.outputFor(trainingVector));
        }
    }
}
