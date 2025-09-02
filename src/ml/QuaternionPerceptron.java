/*
 * Copyright (c) 2025 Hieronim Kubica
 * Licensed under the MIT License. See LICENSE file for full terms.
 */
package ml;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;

import math.Quaternion;

/**
 * A perceptron model that uses quaternions for weights, inputs, and outputs.
 *
 * <p>This implementation uses two quaternion weights, representing rotations, that operate in
 * different coordinate frames:
 *
 * <ul>
 *   <li><strong>Bias rotation</strong> - Applied first as world-frame rotation: biasRotation *
 *       inputOrientation * actionRotation. This rotation is independent of the inputOrientation's
 *       orientation (global coordinate system)
 *   <li><strong>Action rotation</strong> - Applied last as local-frame rotation: biasRotation *
 *       inputOrientation * actionRotation. This rotation is applied relative to the
 *       inputOrientation's transformed orientation
 * </ul>
 *
 * <p>The quaternion chain biasRotation * inputOrientation * actionRotation allows the model to
 * learn both global transformations and input-specific adjustments through separate rotation
 * parameters.
 *
 * <p>The learning rate is adaptive and based on the cross product magnitude between predicted and
 * target orientations, providing automatic convergence as errors approach zero.
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
   * Returns the random seed used for initialization.
   *
   * @return the random seed value
   */
  public long getRandomSeed() {
    return random.nextLong(); // Note: This returns a new random value, not the original seed
  }

  /**
   * Performs the forward pass through the perceptron.
   *
   * <p>Applies the learned rotation to transform the input orientation to the target orientation.
   *
   * @param inputOrientation the input orientation quaternion (must be a unit quaternion)
   * @return the transformed orientation quaternion
   * @throws IllegalArgumentException if input is null or not a unit quaternion
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
   * Classifies an input by converting the predicted quaternion to a binary label.
   *
   * <p>Compares the predicted quaternion to the representations of 0 and 1 target quaternions and
   * assigns the label based on minimum distance.
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
   * Computes the rotation gradient field in a single pass through the batch.
   *
   * @param inputOrientations the input orientations
   * @param predictedOrientations the predicted orientations
   * @param targetOrientations the target orientations
   * @return a GradientFields object containing the rotation gradient
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
   * Solves a 3x3 linear system Ax = b using Apache Commons Math.
   *
   * @param A the coefficient matrix (3x3)
   * @param b the right-hand side vector (3 elements)
   * @return the solution vector x
   * @throws SingularMatrixException if the basis vectors are linearly dependent
   */
  private double[] solveLinearSystem3x3(double[][] A, double[] b) {
    if (A.length != 3 || A[0].length != 3 || b.length != 3) {
      throw new IllegalArgumentException("Matrix A must be 3x3 and vector b must have 3 elements");
    }

    // Create RealMatrix and RealVector from arrays
    RealMatrix matrix = new Array2DRowRealMatrix(A);
    RealVector vector = new ArrayRealVector(b);

    // Use LU decomposition for solving
    DecompositionSolver solver = new LUDecomposition(matrix).getSolver();
    RealVector solution = solver.solve(vector);

    return solution.toArray();
  }

  /**
   * Initializes a random unit quaternion for rotation initialization.
   *
   * <p>Creates a quaternion near the identity (1, 0, 0, 0) with small random perturbation, then
   * normalizes to unit length. This ensures the initial rotations represent small rotations around
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
