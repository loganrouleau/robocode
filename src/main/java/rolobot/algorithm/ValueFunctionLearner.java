package rolobot.algorithm;

import java.io.File;
import java.io.IOException;

public interface ValueFunctionLearner<T> {
    public double outputFor(T actionState);

    public double train(T actionState, double value);

    public void load(File file) throws IOException;

    public void save(File file);
}
