package rolobot.algorithm;

import Jama.Matrix;
import java.io.File;
import java.io.IOException;

public interface ValueFunctionLearner {
    double outputFor(Matrix trainingVector);

    double train(Matrix trainingVector, double targetValue);

    void load(File file) throws IOException;

    void save(File file);
}
