package ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import math.Quaternion;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;

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
 * @see Quaternion
 */
public final class QuaternionPerceptron {

  /** Fixed axis for binary classification */
  private static final double[] CLASSIFICATION_AXIS = {0, 0, 1}; // Z-axis

  /** Learning rate for rotation updates */
  private final double learningRate;

  /** Random number generator for rotation initialization */
  private final Random random;

  /**
   * Bias rotation - Applied first as world-frame rotation: biasRotation * inputOrientation *
   * actionRotation
   */
  private Quaternion biasRotation;

  /**
   * Action rotation - Applied last as local-frame rotation: biasRotation * inputOrientation *
   * actionRotation
   */
  private Quaternion actionRotation;

  // Constructors
  /**
   * Constructs a QuaternionPerceptron with the specified parameters.
   *
   * @param learningRate the learning rate for rotation updates (must be positive)
   * @param randomSeed the seed for reproducible rotation initialization
   * @throws IllegalArgumentException if learningRate is not positive
   */
  public QuaternionPerceptron(double learningRate, long randomSeed) {
    if (learningRate <= 0.0) {
      throw new IllegalArgumentException("Learning rate must be positive, got: " + learningRate);
    }

    this.learningRate = learningRate;
    this.random = new Random(randomSeed);

    // Initialize rotations as random unit quaternions
    this.biasRotation = initializeRandomUnitQuaternion();
    this.actionRotation = initializeRandomUnitQuaternion();
  }

  /** Constructs a QuaternionPerceptron with default learning rate and random seed. */
  public QuaternionPerceptron() {
    this(0.01, System.currentTimeMillis());
  }

  // Public methods
  /**
   * Returns the current learning rate.
   *
   * @return the learning rate value
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * Returns the current bias rotation.
   *
   * <p>The biasRotation represents a world-frame rotation that is applied first in the quaternion
   * chain biasRotation * inputOrientation * actionRotation. This rotation is independent of the
   * inputOrientation.
   *
   * @return the biasRotation quaternion
   */
  public Quaternion getBiasRotation() {
    return biasRotation;
  }

  /**
   * Returns the current action rotation.
   *
   * <p>The actionRotation represents a local-frame rotation that is applied last in the quaternion
   * chain biasRotation * inputOrientation * actionRotation. This rotation is applied relative to
   * the transformed inputOrientation.
   *
   * @return the actionRotation quaternion
   */
  public Quaternion getActionRotation() {
    return actionRotation;
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
   * <p>Applies the quaternion chain: biasRotation * inputOrientation * actionRotation, where:
   *
   * <ul>
   *   <li><strong>Bias rotation</strong> - Applied first as world-frame rotation
   *   <li><strong>Input orientation</strong> - The input orientation quaternion
   *   <li><strong>Action rotation</strong> - Applied last as local-frame rotation
   * </ul>
   *
   * <p>This method assumes the input is a single quaternion representing an orientation.
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

    // Apply the quaternion chain: biasRotation * inputOrientation * actionRotation
    // This represents the composition of three rotations
    Quaternion intermediate = biasRotation.multiply(inputOrientation);
    Quaternion output = intermediate.multiply(actionRotation);

    // The result should automatically be a unit quaternion since:
    // - biasRotation is unit (initialized and maintained as unit)
    // - inputOrientation is unit (validated above)
    // - actionRotation is unit (initialized and maintained as unit)
    // - Quaternion multiplication preserves unit length
    return output;
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
   * Computes action and bias gradient fields in a single pass through the batch.
   *
   * @param inputOrientations the input orientations
   * @param predictedOrientations the predicted orientations
   * @param targetOrientations the target orientations
   * @return a GradientFields object containing both bias and action gradients
   */
  public GradientFields computeGradientFields(
      List<Quaternion> inputOrientations,
      List<Quaternion> predictedOrientations,
      List<Quaternion> targetOrientations) {
    // Collect individual rotation updates from each sample
    List<Quaternion> biasUpdates = new ArrayList<>();
    List<Quaternion> actionUpdates = new ArrayList<>();

    for (int i = 0; i < inputOrientations.size(); i++) {
      Quaternion input = inputOrientations.get(i);
      Quaternion predicted = predictedOrientations.get(i);
      Quaternion target = targetOrientations.get(i);

      // Compute error rotation: rotation from predicted to target
      Quaternion errorRotation = predicted.geodesicRotation(target);

      // Decompose error into bias, residual, and action components
      DecompositionResult decomposition = decomposeUpdate(errorRotation, input);

      // Collect individual updates for proper aggregation
      biasUpdates.add(decomposition.biasUpdate);
      actionUpdates.add(decomposition.actionUpdate);
    }

    // Aggregate rotations using exponential map summation
    Quaternion aggregatedBiasGradient = exponentialMapSum(biasUpdates);
    Quaternion aggregatedActionGradient = exponentialMapSum(actionUpdates);

    return new GradientFields(aggregatedBiasGradient, aggregatedActionGradient);
  }

  /**
   * Updates the rotations using the computed gradient fields via exponential map.
   *
   * @param gradientFields the gradient fields for both bias and action rotations
   */
  private void updateRotations(GradientFields gradientFields) {
    // TODO: Implement exponential map update
    // biasRotation *= exp(learningRate * gradientFields.biasGradient)
    // actionRotation *= exp(learningRate * gradientFields.actionGradient)
  }

  /**
   * Aggregates a list of rotations using exponential map summation. This method converts rotations
   * to rotation vectors (tangent space), sums them, then maps back to the quaternion group via
   * exponential map.
   *
   * @param rotations the list of rotations to aggregate
   * @return the summed rotation quaternion
   */
  private Quaternion exponentialMapSum(List<Quaternion> rotations) {
    if (rotations.isEmpty()) {
      return Quaternion.ONE;
    }
    if (rotations.size() == 1) {
      return rotations.get(0);
    }

    // Convert rotations to rotation vectors (tangent space)
    double[] avgVector = {0.0, 0.0, 0.0};
    for (Quaternion q : rotations) {
      double[] v = q.toRotationVector();
      for (int i = 0; i < 3; i++) {
        avgVector[i] += v[i];
      }
    }

    // Sum the rotation vectors (no division by batch size)
    // This maintains the magnitude scaling with batch size, similar to gradient accumulation

    // Map back to quaternion group via exponential map
    return Quaternion.fromRotationVector(avgVector);
  }

  /**
   * Decomposes the error rotation into bias, residual, and action components. This is the full
   * implementation matching the Python decompose_update method.
   *
   * @param errorRotation the error rotation to decompose
   * @param input the input orientation
   * @return a DecompositionResult containing the three update components
   */
  private DecompositionResult decomposeUpdate(Quaternion errorRotation, Quaternion input) {
    // Convert quaternions to rotation vectors (3D vectors)
    double[] vBias = biasRotation.toRotationVector();
    double[] vInput = input.toRotationVector();
    double[] vAction = actionRotation.toRotationVector();
    double[] vError = errorRotation.toRotationVector();

    // Form a basis from the vectors v_b, v_k, and v_a
    double[][] basis = {vBias, vInput, vAction};

    // Check if basis is linearly independent and solve for coefficients
    double[] coefficients = solveLinearSystem3x3(basis, vError);

    // Project error components back into quaternion space using quaternion exponentiation
    Quaternion biasUpdate = computeBiasUpdate(coefficients, input, errorRotation);
    Quaternion residualUpdate = computeResidualUpdate(coefficients, errorRotation);
    Quaternion actionUpdate = computeActionUpdate(coefficients, input, errorRotation);

    return new DecompositionResult(biasUpdate, residualUpdate, actionUpdate);
  }

  /** Computes the bias update component from the decomposition coefficients. */
  private Quaternion computeBiasUpdate(
      double[] coefficients, Quaternion input, Quaternion errorRotation) {
    // u_b = (q_kernel.conjugate() ** coefficients[1] *
    // self.action.conjugate() ** coefficients[2] *
    // q_update ** coefficients[0])
    Quaternion term1 = input.conjugate().pow(coefficients[1]);
    Quaternion term2 = actionRotation.conjugate().pow(coefficients[2]);
    Quaternion term3 = errorRotation.pow(coefficients[0]);

    Quaternion result = term1.multiply(term2).multiply(term3);
    return result.normalize();
  }

  /** Computes the residual update component from the decomposition coefficients. */
  private Quaternion computeResidualUpdate(double[] coefficients, Quaternion errorRotation) {
    // u_residual = (self.bias.conjugate() ** coefficients[0] *
    // self.action.conjugate() ** coefficients[2] *
    // q_update ** coefficients[1])
    Quaternion term1 = biasRotation.conjugate().pow(coefficients[0]);
    Quaternion term2 = actionRotation.conjugate().pow(coefficients[2]);
    Quaternion term3 = errorRotation.pow(coefficients[1]);

    Quaternion result = term1.multiply(term2).multiply(term3);
    return result.normalize();
  }

  /** Computes the action update component from the decomposition coefficients. */
  private Quaternion computeActionUpdate(
      double[] coefficients, Quaternion input, Quaternion errorRotation) {
    // u_a = (self.bias.conjugate() ** coefficients[0] *
    // q_kernel.conjugate() ** coefficients[1] *
    // q_update ** coefficients[2])
    Quaternion term1 = biasRotation.conjugate().pow(coefficients[0]);
    Quaternion term2 = input.conjugate().pow(coefficients[1]);
    Quaternion term3 = errorRotation.pow(coefficients[2]);

    Quaternion result = term1.multiply(term2).multiply(term3);
    return result.normalize();
  }

  /** Result container for the decomposition of an error rotation. */
  private static class DecompositionResult {
    final Quaternion biasUpdate;
    final Quaternion residualUpdate;
    final Quaternion actionUpdate;

    DecompositionResult(Quaternion biasUpdate, Quaternion residualUpdate, Quaternion actionUpdate) {
      this.biasUpdate = biasUpdate;
      this.residualUpdate = residualUpdate;
      this.actionUpdate = actionUpdate;
    }
  }

  /** Result container for gradient fields computed from a batch. */
  public static class GradientFields {
    public final Quaternion biasGradient;
    public final Quaternion actionGradient;

    public GradientFields(Quaternion biasGradient, Quaternion actionGradient) {
      this.biasGradient = biasGradient;
      this.actionGradient = actionGradient;
    }
  }

  /**
   * Solves a 3x3 linear system Ax = b using Apache Commons Math.
   *
   * @param A the coefficient matrix (3x3)
   * @param b the right-hand side vector (3 elements)
   * @return the solution vector x
   */
  private double[] solveLinearSystem3x3(double[][] A, double[] b) {
    if (A.length != 3 || A[0].length != 3 || b.length != 3) {
      throw new IllegalArgumentException("Matrix A must be 3x3 and vector b must have 3 elements");
    }

    try {
      // Create RealMatrix and RealVector from arrays
      RealMatrix matrix = new Array2DRowRealMatrix(A);
      RealVector vector = new ArrayRealVector(b);

      // Use LU decomposition for solving
      DecompositionSolver solver = new LUDecomposition(matrix).getSolver();
      RealVector solution = solver.solve(vector);

      return solution.toArray();
    } catch (SingularMatrixException e) {
      throw new IllegalStateException(
          "Basis vectors are linearly dependent: " + Arrays.deepToString(A), e);
    }
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
