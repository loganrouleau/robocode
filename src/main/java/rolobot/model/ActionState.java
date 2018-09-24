package rolobot.model;

import Jama.Matrix;
import robocode.Robot;
import robocode.ScannedRobotEvent;

public class ActionState {
    private int x;
    private int y;
    private int enemyX;
    private int enemyY;
    private Action action;

    public ActionState() {}

    public ActionState(int x, int y, int enemyX, int enemyY, Action action) {
        this.x = x;
        this.y = y;
        this.enemyX = enemyX;
        this.enemyY = enemyY;
        this.action = action;
    }

    public void updateState(Robot bot, ScannedRobotEvent event) {
        x = (int) bot.getX();
        y = (int) bot.getY();

        double angle = Math.toRadians(bot.getHeading() + (event.getBearing() % 360));
        enemyX = (int) ((Math.round((x + Math.sin(angle) * event.getDistance()) / 200) * 200) / 200);
        enemyY = (int) ((Math.round((y + Math.cos(angle) * event.getDistance()) / 200) * 200) / 200);

        if (x < 95.5) {
            x = 0;
        } else if (x < 304.5) {
            x = 1;
        } else if (x < 495.5) {
            x = 2;
        } else if (x < 686.5) {
            x = 3;
        } else {
            x = 4;
        }
        if (y < 159) {
            y = 0;
        } else if (y < 441) {
            y = 1;
        } else {
            y = 2;
        }
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getEnemyX() {
        return enemyX;
    }

    public int getEnemyY() {
        return enemyY;
    }

    public enum Action {
        UP(0, 'U'),
        DOWN(1, 'D'),
        LEFT(2, 'L'),
        RIGHT(3, 'R'),
        FIRE(4, 'F');

        private final int label;
        private final char shortName;

        Action(int label, char shortName) {
            this.label = label;
            this.shortName = shortName;
        }

        public int label() {
            return label;
        }

        public char shortName() {
            return shortName;
        }
    }

    public Action getAction() {
        return action;
    }

    public void setAction(Action action) {
        this.action = action;
    }

    public Matrix asMatrix() {
        return new Matrix(new double[][]{{x, y, enemyX, enemyY, 1}});
    }

    @Override
    public String toString() {
        return "ActionState{" +
                "x=" + x +
                ", y=" + y +
                ", enemyX=" + enemyX +
                ", enemyY=" + enemyY +
                ", action=" + action +
                '}';
    }

    /*@Override
    public int compareTo(ActionState o) {
        return 0;
    }*/

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ActionState that = (ActionState) o;

        if (getX() != that.getX()) return false;
        if (getY() != that.getY()) return false;
        if (getEnemyX() != that.getEnemyX()) return false;
        if (getEnemyY() != that.getEnemyY()) return false;
        return getAction() == that.getAction();
    }

    @Override
    public int hashCode() {
        int result = getX();
        result = 31 * result + getY();
        result = 31 * result + getEnemyX();
        result = 31 * result + getEnemyY();
        result = 31 * result + getAction().hashCode();
        return result;
    }
}
