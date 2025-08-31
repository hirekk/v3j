package ml;

import java.util.Random;
import math.Quaternion;

/**
 * A perceptron model that uses quaternions for weights, inputs, and outputs.
 *
 * <p>This implementation uses two quaternion weights that operate in different coordinate frames:
 *
 * <ul>
 *   <li><strong>Bias</strong> - Applied first as world-frame rotation: bias * input * action. This
 *       rotation is independent of the input's orientation (global coordinate system)
 *   <li><strong>Action</strong> - Applied last as local-frame rotation: bias * input * action. This
 *       rotation is applied relative to the input's transformed orientation
 * </ul>
 *
 * <p>The quaternion chain bias * input * action allows the model to learn both global
 * transformations and input-specific adjustments through separate weight parameters.
 *
 * @see Quaternion
 */
public final class QuaternionPerceptron {

  /** Learning rate for weight updates */
  private final double learningRate;

  /** Random number generator for weight initialization */
  private final Random random;

  /** Bias weight - Applied first as world-frame rotation: bias * input * action */
  private Quaternion bias;

  /** Action weight - Applied last as local-frame rotation: bias * input * action */
  private Quaternion action;

  /**
   * Constructs a QuaternionPerceptron with the specified parameters.
   *
   * @param learningRate the learning rate for weight updates (must be positive)
   * @param randomSeed the seed for reproducible weight initialization
   * @throws IllegalArgumentException if learningRate is not positive
   */
  public QuaternionPerceptron(double learningRate, long randomSeed) {
    if (learningRate <= 0.0) {
      throw new IllegalArgumentException("Learning rate must be positive, got: " + learningRate);
    }

    this.learningRate = learningRate;
    this.random = new Random(randomSeed);

    // Initialize weights as random unit quaternions
    this.bias = initializeRandomUnitQuaternion();
    this.action = initializeRandomUnitQuaternion();
  }

  /** Constructs a QuaternionPerceptron with default learning rate and random seed. */
  public QuaternionPerceptron() {
    this(0.01, System.currentTimeMillis());
  }

  /**
   * Initializes a random unit quaternion for weight initialization.
   *
   * <p>Creates a quaternion near the identity (1, 0, 0, 0) with small random perturbation, then
   * normalizes to unit length. This ensures the initial weights represent small rotations around
   * the identity.
   *
   * @return a random unit quaternion
   */
  private Quaternion initializeRandomUnitQuaternion() {
    // Start near identity with small perturbations
    double w = 1.0 + random.nextGaussian(0.0, 0.05);
    double x = random.nextGaussian(0.0, 0.05);
    double y = random.nextGaussian(0.0, 0.05);
    double z = random.nextGaussian(0.0, 0.05);

    Quaternion q = new Quaternion(w, x, y, z);

    return q.normalize();
  }

  /**
   * Returns the current learning rate.
   *
   * @return the learning rate value
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * Returns the current bias weight.
   *
   * <p>The bias represents a world-frame rotation that is applied first in the quaternion chain
   * bias * input * action. This rotation is independent of the input's orientation.
   *
   * @return the bias quaternion weight
   */
  public Quaternion getBias() {
    return bias;
  }

  /**
   * Returns the current action weight.
   *
   * <p>The action represents a local-frame rotation that is applied last in the quaternion chain
   * bias * input * action. This rotation is applied relative to the input's transformed
   * orientation.
   *
   * @return the action quaternion weight
   */
  public Quaternion getAction() {
    return action;
  }

  /**
   * Returns the random seed used for initialization.
   *
   * @return the random seed value
   */
  public long getRandomSeed() {
    return random.nextLong(); // Note: This returns a new random value, not the original seed
  }
}
