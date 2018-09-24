package rolobot;

import static rolobot.Config.ACTIONS;
import static rolobot.Config.ALPHA;
import static rolobot.Config.CYCLES;
import static rolobot.Config.EPSILON_INITIAL;
import static rolobot.Config.GAMMA;
import static rolobot.Config.NUM_ENEMY_X_STATES;
import static rolobot.Config.NUM_ENEMY_Y_STATES;
import static rolobot.Config.NUM_X_STATES;
import static rolobot.Config.NUM_Y_STATES;
import static rolobot.Config.RESET_LUT;
import static rolobot.Config.SAVE_LUT;
import static rolobot.Config.SAVE_NET;
import static rolobot.Config.TESTING_ROUNDS;
import static rolobot.Config.TRAINING_ROUNDS;
import static rolobot.Config.TRAIN_NET;
import static rolobot.model.ActionState.Action.UP;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.RobocodeFileOutputStream;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;
import rolobot.algorithm.NeuralNetwork;
import rolobot.model.ActionState;

public class RoloBot extends AdvancedRobot {
    private static File LUTpath;
    private static File Infopath;

    // Q-learning parameters
    private static double epsilon = EPSILON_INITIAL;

    // Configure training
    private static int reward, wins;

    // Each round causes a new RoloBot to be instantiated, so we use static fields to persist data
    private static Random random = new Random();
    private static Map<ActionState, Double> lookupTable;
    private static NeuralNetwork neuralNet;
    private static int totalReward = 0;
    private static int[] cumulativeReward = new int[TRAINING_ROUNDS * CYCLES];
    private static int cumRewardIndex = 0;
    private static int[] trainingWins = new int[CYCLES];
    private static int[] testingWins = new int[CYCLES];
    private static int round;
    private static int cycle;
    private static int trainingEnd = TRAINING_ROUNDS - 1;

    private ActionState actionState = new ActionState();
    private boolean rewardReceived = false;

    @Override
    public void run() {
        round = getRoundNum();

        // Some things can only be done inside run()
        if (round == 0) {
            LUTpath = getDataFile("lookup-table.dat");
            Infopath = getDataFile("info.dat");
            neuralNet = new NeuralNetwork();
            neuralNet.neuralNetFile = getDataFile("neural-network.dat");
        }
        lookupTable = readLUT();

        // Customize robot
        if (epsilon > 0) {
            setBodyColor(Color.red);
        } else {
            setBodyColor(Color.green);
        }
        setGunColor(Color.black);
        setScanColor(Color.yellow);
        while (true) {
            // Return to 5x3 grid if necessary
            double heading = this.getHeading();
            double error;
            if ((Math.round(this.getX()) - 18) % 191 > 0.1) {
                System.out.println("Correcting x position");
                // Move left to correct x position
                error = (this.getX() - 18) % 191;

                if (heading > 270) {
                    turnLeft(heading - 270);
                    ahead(error);
                } else if (heading < 90) {
                    turnLeft(heading + 90);
                    ahead(error);
                } else {
                    turnRight(270 - heading);
                    ahead(error);
                }
            }
            if ((Math.round(this.getY()) - 18) % 282 > 0.1) {
                System.out.println("Correcting y position");
                // Move down to correct y position
                error = (this.getY() - 18) % 282;
                if (heading > 180) {
                    turnLeft(heading - 180);
                    ahead(error);
                } else {
                    turnRight(180 - heading);
                    ahead(error);
                }
            }
            // Scan until enemy sighted
            turnLeft(360);
        }
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
        ActionState prevActionState = actionState;
        actionState.updateState(this, event);

        // Choose action based on policy
        // Greedy action
        ActionState.Action action = UP;
        double Qmax = -Double.MAX_VALUE;
        for (ActionState.Action possibleAction : ActionState.Action.values()) {
            actionState.setAction(possibleAction);
            if (TRAIN_NET) {
                if (neuralNet.outputFor(actionState.asMatrix()) > Qmax) {
                    Qmax = neuralNet.outputFor(actionState.asMatrix());
                    action = possibleAction;
                }
            } else {
                if (lookupTable.get(actionState) > Qmax) {
                    Qmax = lookupTable.get(actionState);
                    action = possibleAction;
                }
            }
        }

        // Random action
        if (random.nextDouble() < epsilon) {
            System.out.println("Choosing random action");
            action = ActionState.Action.values()[random.nextInt(ACTIONS)];
        }

        actionState.setAction(action);
        System.out.println(actionState);

        // Update lookupTable or neural net if a reward was received since previous action
        if (rewardReceived) {
            if (TRAIN_NET) {
                double qprev = neuralNet.outputFor(prevActionState.asMatrix());
                qprev += ALPHA * (reward + GAMMA * Qmax - qprev);
                neuralNet.train(prevActionState.asMatrix(), qprev);
            } else {
                double prevValue = lookupTable.get(prevActionState);
                lookupTable.put(prevActionState, prevValue += ALPHA * (reward + GAMMA * Qmax - prevValue));
            }
            totalReward += reward;
            reward = 0;
            rewardReceived = false;
        }

        performAction(action);
    }

    private void performAction(ActionState.Action action) {
        double heading = this.getHeading();
        switch (action) {
            case UP:
                if (heading > 180) {
                    turnRight(360 - heading);
                    ahead(282);
                } else {
                    turnLeft(heading);
                    ahead(282);
                }
                break;
            case DOWN:
                if (heading > 180) {
                    turnLeft(heading - 180);
                    ahead(282);
                } else {
                    turnRight(180 - heading);
                    ahead(282);
                }
                break;
            case LEFT:
                if (heading > 270) {
                    turnLeft(heading - 270);
                    ahead(191);
                } else if (heading < 90) {
                    turnLeft(heading + 90);
                    ahead(191);
                } else {
                    turnRight(270 - heading);
                    ahead(191);
                }
                break;
            case RIGHT:
                if (heading > 270) {
                    turnRight(450 - heading);
                    ahead(191);
                } else if (heading < 90) {
                    turnRight(90 - heading);
                    ahead(191);
                } else {
                    turnLeft(heading - 90);
                    ahead(191);
                }
                break;
            case FIRE:
                fire(3);
                break;
            default:
                throw new RuntimeException("Unreachable");
        }
    }

    @Override
    public void onWin(WinEvent event) {
        if (epsilon > 0) {
            reward += 100;
            rewardReceived = true;
        }
        wins += 1;
    }

    @Override
    public void onDeath(DeathEvent event) {
        if (epsilon > 0) {
            reward -= 50;
            rewardReceived = true;
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        if (epsilon > 0) {
            reward -= 3;
            rewardReceived = true;
            System.out.println("reward: " + reward);
        }
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        if (epsilon > 0) {
            reward += 1;
            rewardReceived = true;
            System.out.println("reward: " + reward);
        }
    }

    @Override
    public void onBulletMissed(BulletMissedEvent event) {
        if (epsilon > 0) {
            reward -= 1;
            rewardReceived = true;
            System.out.println("reward: " + reward);
        }
    }

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        if (epsilon > 0) {
            cumulativeReward[cumRewardIndex] = totalReward;
            //bias_out[cumRewardIndex] = w[0];
            cumRewardIndex += 1;
        }
        if (round == trainingEnd) {
            trainingWins[cycle] = wins;
            wins = 0;
            epsilon = 0;
        }

        if (round == trainingEnd + TESTING_ROUNDS) {
            testingWins[cycle] = wins;
            wins = 0;
            cycle += 1;
            trainingEnd += TRAINING_ROUNDS + TESTING_ROUNDS;
            epsilon = EPSILON_INITIAL;
        }
    }

    @Override
    public void onBattleEnded(BattleEndedEvent event) {
        if (SAVE_LUT) {
            writeLUT();
        }
        if (SAVE_NET) {
            neuralNet.initializeWeights(15);
            neuralNet.save(neuralNet.neuralNetFile);
        }
        writeInfo();
    }

    private void writeInfo() {
        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(Infopath));
            w.println("Training Testing");
            for (int i = 0; i < CYCLES; i++) {
                w.printf("%d %d%n", trainingWins[i], testingWins[i]);
            }
            w.println("Cumulative_Reward");
            for (int i = 0; i < TRAINING_ROUNDS * CYCLES; i++) {
                w.printf("%d%n", cumulativeReward[i]);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error writing to info file");
        } finally {
            if (w != null) {
                w.close();
            }
        }
    }

    private static Map<ActionState, Double> readLUT() {
        Map<ActionState, Double> result = new HashMap<>();
        if (RESET_LUT) {
            for (int i = 0; i < NUM_X_STATES; i++) {
                for (int j = 0; j < NUM_Y_STATES; j++) {
                    for (int k = 0; k < NUM_ENEMY_X_STATES; k++) {
                        for (int l = 0; l < NUM_ENEMY_Y_STATES; l++) {
                            for (ActionState.Action action : ActionState.Action.values()) {
                                ActionState actionState = new ActionState(i, j, k, l, action);
                                System.out.println("Reading from LUT: " + actionState);
                                result.put(actionState, random.nextDouble() / 100);
                            }
                        }
                    }
                }
            }
        } else {
            try {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(new FileReader(LUTpath));
                    String sCurrentLine;
                    int x = 0;
                    int y = 0;
                    int enx = 0;
                    int eny = 0;
                    double q = 0;
                    while ((sCurrentLine = reader.readLine()) != null) {
                        String[] arr = sCurrentLine.split(" ");
                        x = Integer.parseInt(arr[0]);
                        y = Integer.parseInt(arr[1]);
                        enx = Integer.parseInt(arr[2]);
                        eny = Integer.parseInt(arr[3]);
                        char a = arr[4].charAt(0);
                        q = Double.parseDouble(arr[5]);
                        ActionState.Action action = Arrays.stream(ActionState.Action.values()).filter(e -> e.shortName() == a).findFirst().orElseThrow(() -> new IllegalStateException("Can't find action"));
                        result.put(new ActionState(x, y, enx, eny, action), q);
                    }
                } finally {
                    if (reader != null) {
                        reader.close();
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException("Error reading from LUT");
            }
        }
        return result;
    }

    private void writeLUT() {
        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(LUTpath));
            for (int i = 0; i < NUM_X_STATES; i++) {
                for (int j = 0; j < NUM_Y_STATES; j++) {
                    for (int k = 0; k < NUM_ENEMY_X_STATES; k++) {
                        for (int l = 0; l < NUM_ENEMY_Y_STATES; l++) {
                            for (ActionState.Action action : ActionState.Action.values()) {
                                w.printf("%d %d %d %d %c %.5f%n", i, j, k, l, action.shortName(), lookupTable.get(new ActionState(i, j, k, l, action)));
                            }
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Error writing LUT");
        } finally {
            if (w != null) {
                w.close();
            }
        }
    }
}
