package rolobot;

public final class Config {
    private Config() {}

    public static final boolean RESET_LUT = false;
    public static final boolean SAVE_LUT = true;

    public static final boolean TRAIN_NET = false;
    public static final boolean RESET_NET = true;
    public static final boolean SAVE_NET = true;
    public static final boolean USE_BIPOLAR = true;

    public static final double ALPHA = 0.1;
    public static final double GAMMA = 0.9;
    public static final double EPSILON_INITIAL = 0.15;

    // Neural net parameters
    public static final double STEP_SIZE = 0.0001;
    public static final double ALPHA_NET = 0.9;
    public static final int INPUT_NODES = 9;
    public static final int HIDDEN_NODES = 25;
    public static final double QSCALE = 50;

    public static final int NUM_X_STATES = 5;
    public static final int NUM_Y_STATES = 3;
    public static final int NUM_ENEMY_X_STATES = 5;
    public static final int NUM_ENEMY_Y_STATES = 4;
    public static final int ACTIONS = 5;

    public static final int TESTING_ROUNDS = 10;
    public static final int TRAINING_ROUNDS = 90;
    public static final int CYCLES = 20;
}
