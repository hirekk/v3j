/*
 * Copyright (c) 2025 Hieronim Kubica
 * Licensed under the MIT License. See LICENSE file for full terms.
 */
package ml;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import math.Quaternion;

/**
 * Quaternion-based perceptron for binary classification of 3D orientations.
 *
 * <p>Uses a single rotation weight to transform input orientations. The learning rate is adaptive
 * based on error magnitude for automatic convergence.
 *
 * @see Quaternion
 */
public final class QuaternionPerceptron {

  /** Fixed axis for binary classification */
  private static final double[] CLASSIFICATION_AXIS = {0, 0, 1}; // Z-axis

  /** Random number generator for rotation initialization */
  private final Random random;

  /** Single rotation weight that transforms input orientations to target orientations */
  private Quaternion rotation;

  // Constructors
  /**
   * Constructs a QuaternionPerceptron with the specified parameters.
   *
   * @param randomSeed the seed for reproducible rotation initialization
   */
  public QuaternionPerceptron(long randomSeed) {
    this.random = new Random(randomSeed);

    // Initialize rotation as random unit quaternion
    this.rotation = initializeRandomUnitQuaternion();
  }

  /** Constructs a QuaternionPerceptron with default random seed. */
  public QuaternionPerceptron() {
    this(System.currentTimeMillis());
  }

  // Public methods

  /**
   * Returns the current rotation weight.
   *
   * <p>The rotation represents the learned transformation that maps input orientations to target
   * orientations.
   *
   * @return the rotation quaternion
   */
  public Quaternion getRotation() {
    return rotation;
  }

  /**
   * Returns a random long value from the internal random generator.
   *
   * @return a random long value
   */
  public long getRandomSeed() {
    return random.nextLong();
  }

  /**
   * Performs the forward pass: rotation * inputOrientation.
   *
   * @param inputOrientation the input orientation quaternion (must be unit)
   * @return the transformed orientation quaternion
   * @throws IllegalArgumentException if input is null or not unit
   */
  public Quaternion forward(Quaternion inputOrientation) {
    if (inputOrientation == null) {
      throw new IllegalArgumentException("Input quaternion cannot be null");
    }

    if (!inputOrientation.isUnit()) {
      throw new IllegalArgumentException("Input quaternion must be a unit quaternion");
    }

    // Apply the learned rotation: rotation * inputOrientation
    // This represents the transformation from input to target orientation
    Quaternion output = rotation.multiply(inputOrientation);

    // The result should automatically be a unit quaternion since:
    // - rotation is unit (initialized and maintained as unit)
    // - inputOrientation is unit (validated above)
    // - Quaternion multiplication preserves unit length
    return output;
  }

  /**
   * Classifies input by comparing predicted quaternion to target representations.
   *
   * @param inputOrientation the input orientation quaternion
   * @return the predicted binary label (0 or 1)
   */
  public int classify(Quaternion inputOrientation) {
    Quaternion predicted = forward(inputOrientation);

    // Use existing method to get target quaternion representations
    Quaternion target0 = labelToTargetOrientation(0);
    Quaternion target1 = labelToTargetOrientation(1);

    // Compute geodesic distances along great circles on S³
    // The geodesic distance is the angle between quaternions
    double distanceTo0 = predicted.geodesicDistance(target0);
    double distanceTo1 = predicted.geodesicDistance(target1);

    // Return the label with minimum distance
    return (distanceTo0 <= distanceTo1) ? 0 : 1;
  }

  /**
   * Performs a single training step using the provided batch of inputs and targets.
   *
   * @param inputOrientations the input orientations
   * @param binaryLabels the sample labels
   */
  public void step(List<Quaternion> inputOrientations, List<Integer> binaryLabels) {
    if (inputOrientations == null || binaryLabels == null) {
      throw new IllegalArgumentException("Inputs and labels cannot be null");
    }
    if (inputOrientations.size() != binaryLabels.size()) {
      throw new IllegalArgumentException("Input and label lists must have same size");
    }

    List<Quaternion> targetOrientations =
        binaryLabels.stream().map(this::labelToTargetOrientation).collect(Collectors.toList());

    List<Quaternion> predictedOrientations =
        inputOrientations.stream().map(this::forward).collect(Collectors.toList());

    GradientFields gradientFields =
        computeGradientFields(inputOrientations, predictedOrientations, targetOrientations);

    updateRotations(gradientFields);
  }

  /**
   * Computes rotation gradient field from batch of inputs, predictions, and targets.
   *
   * @param inputOrientations the input orientations
   * @param predictedOrientations the predicted orientations
   * @param targetOrientations the target orientations
   * @return gradient fields containing the rotation gradient
   */
  public GradientFields computeGradientFields(
      List<Quaternion> inputOrientations,
      List<Quaternion> predictedOrientations,
      List<Quaternion> targetOrientations) {

    // Accumulate gradient across the batch
    double[] rotationGradientVector = {0.0, 0.0, 0.0};

    for (int i = 0; i < inputOrientations.size(); i++) {
      Quaternion input = inputOrientations.get(i);
      Quaternion predicted = predictedOrientations.get(i);
      Quaternion target = targetOrientations.get(i);

      // Compute geodesic error: the rotation needed to go from predicted to target
      // This is: target * predicted^(-1)
      Quaternion error = target.multiply(predicted.inverse());

      // For unit quaternions, the error magnitude represents the rotation angle
      // We can use the geodesic distance as the adaptive rate
      double adaptiveRate = predicted.geodesicDistance(target);

      // Skip if error is too small (quaternions nearly coincide)
      if (adaptiveRate < 1e-6) {
        continue;
      }

      // Convert error to rotation vector in tangent space
      double[] vError = error.log().getVector();

      // Scale by adaptive rate and accumulate
      // Use smaller learning rate for stability
      double learningRate = 0.003 * adaptiveRate;
      for (int j = 0; j < 3; j++) {
        rotationGradientVector[j] += vError[j] * learningRate;
      }
    }

    // Create gradient quaternion from accumulated vector
    Quaternion rotationGradient = Quaternion.fromRotationVector(rotationGradientVector);

    return new GradientFields(rotationGradient, rotationGradient);
  }

  // TODO: early stopping
  /**
   * Updates the rotation using the computed gradient field via exponential map.
   *
   * @param gradientFields the gradient fields containing the rotation gradient
   */
  private void updateRotations(GradientFields gradientFields) {
    // Update rotation using the gradient
    // Use inverse of gradient to move in opposite direction (gradient descent)
    double[] rotationVector = gradientFields.rotationGradient.toRotationVector();
    Quaternion rotationUpdate = Quaternion.fromRotationVector(rotationVector).inverse();
    rotation = rotationUpdate.multiply(rotation);
  }

  /** Result container for gradient fields computed from a batch. */
  public static class GradientFields {
    public final Quaternion rotationGradient;

    public GradientFields(Quaternion rotationGradient, Quaternion unused) {
      this.rotationGradient = rotationGradient;
      // Keep second parameter for backward compatibility during refactoring
    }
  }

  /**
   * Initializes a random unit quaternion near identity with small perturbations.
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
   * Converts binary label to target orientation.
   *
   * @param label the binary label (0 or 1)
   * @return the target orientation quaternion
   * @throws IllegalArgumentException if label is not 0 or 1
   */
  private Quaternion labelToTargetOrientation(int label) {
    if (label == 0) {
      return Quaternion.ONE; // Target: identity orientation
    } else if (label == 1) {
      return Quaternion.fromAxisAngle(Math.PI, CLASSIFICATION_AXIS); // Target: flipped orientation
    } else {
      throw new IllegalArgumentException("Label must be 0 or 1, got: " + label);
    }
  }
}
