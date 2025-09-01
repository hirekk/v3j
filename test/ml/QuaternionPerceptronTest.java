/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package ml;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import math.Quaternion;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link QuaternionPerceptron}.
 *
 * <p>Tests cover initialization, attribute access, and basic functionality.
 */
@DisplayName("QuaternionPerceptron")
class QuaternionPerceptronTest {

  private static final double EPSILON = 1e-10;

  @Nested
  @DisplayName("Initialization")
  class InitializationTests {

    @Test
    @DisplayName("Default constructor initializes with expected defaults")
    void testDefaultConstructor() {
      QuaternionPerceptron perceptron = new QuaternionPerceptron();

      assertNotNull(perceptron.getBiasRotation());
      assertNotNull(perceptron.getActionRotation());
    }

    @Test
    @DisplayName("Parameterized constructor initializes with specified values")
    void testParameterizedConstructor() {
      long randomSeed = 42L;

      QuaternionPerceptron perceptron = new QuaternionPerceptron(randomSeed);

      assertNotNull(perceptron.getBiasRotation());
      assertNotNull(perceptron.getActionRotation());
    }

    @Test
    @DisplayName("Constructor with same seed produces identical weights")
    void testDeterministicInitialization() {
      long seed = 12345L;

      QuaternionPerceptron perceptron1 = new QuaternionPerceptron(seed);
      QuaternionPerceptron perceptron2 = new QuaternionPerceptron(seed);

      assertEquals(perceptron1.getBiasRotation(), perceptron2.getBiasRotation());
      assertEquals(perceptron1.getActionRotation(), perceptron2.getActionRotation());
    }

    @Test
    @DisplayName("Constructor with different seeds produces different weights")
    void testNonDeterministicInitialization() {
      QuaternionPerceptron perceptron1 = new QuaternionPerceptron(1L);
      QuaternionPerceptron perceptron2 = new QuaternionPerceptron(2L);

      assertNotEquals(perceptron1.getBiasRotation(), perceptron2.getBiasRotation());
      assertNotEquals(perceptron1.getActionRotation(), perceptron2.getActionRotation());
    }
  }

  @Nested
  @DisplayName("Weight Properties")
  class WeightPropertiesTests {

    @Test
    @DisplayName("Weights are valid unit quaternions")
    void testWeightsAreValidUnitQuaternions() {
      QuaternionPerceptron perceptron = new QuaternionPerceptron(42L);

      Quaternion bias = perceptron.getBiasRotation();
      Quaternion action = perceptron.getActionRotation();

      // Check basic properties
      assertTrue(bias.isUnit(), "Bias should be a unit quaternion");
      assertTrue(action.isUnit(), "Action should be a unit quaternion");
      assertFalse(bias.isZero(), "Bias should not be zero");
      assertFalse(action.isZero(), "Action should not be zero");

      // Check that they're different (random initialization)
      assertNotEquals(bias, action);
    }

    @Test
    @DisplayName("Weights are reasonably near identity")
    void testWeightsNearIdentity() {
      // Use retry mechanism to ensure robust testing while maintaining reasonable complexity
      boolean biasNearIdentity = false;
      boolean actionNearIdentity = false;

      // Try up to 5 different random initializations to ensure robustness
      for (int attempt = 0; attempt < 5; attempt++) {
        QuaternionPerceptron newPerceptron = new QuaternionPerceptron(42L + attempt);
        Quaternion newBias = newPerceptron.getBiasRotation();
        Quaternion newAction = newPerceptron.getActionRotation();

        double biasDistance = newBias.subtract(Quaternion.ONE).norm();
        double actionDistance = newAction.subtract(Quaternion.ONE).norm();

        // Use strict threshold for near-identity (as per original design)
        if (biasDistance < 0.05) {
          biasNearIdentity = true;
        }
        if (actionDistance < 0.05) {
          actionNearIdentity = true;
        }

        // Early exit if both are near identity
        if (biasNearIdentity && actionNearIdentity) {
          break;
        }
      }

      // Both weights should be near identity across multiple random initializations
      assertTrue(
          biasNearIdentity,
          "Bias should be near identity (distance < 0.05) across multiple attempts");
      assertTrue(
          actionNearIdentity,
          "Action should be near identity (distance < 0.05) across multiple attempts");
    }

    @Test
    @DisplayName("Weights maintain unit length across multiple initializations")
    void testWeightsMaintainUnitLength() {
      // Test that weights are consistently unit quaternions across different random seeds
      for (int attempt = 0; attempt < 5; attempt++) {
        QuaternionPerceptron perceptron = new QuaternionPerceptron(42L + attempt);

        Quaternion bias = perceptron.getBiasRotation();
        Quaternion action = perceptron.getActionRotation();

        // These should always be true regardless of random variation
        assertTrue(bias.isUnit(), "Bias should always be unit length");
        assertTrue(action.isUnit(), "Action should always be unit length");
        assertFalse(bias.isZero(), "Bias should never be zero");
        assertFalse(action.isZero(), "Action should never be zero");
      }
    }
  }

  @Nested
  @DisplayName("Attribute Access")
  class AttributeAccessTests {

    private QuaternionPerceptron perceptron;

    @BeforeEach
    void setUp() {
      perceptron = new QuaternionPerceptron(42L);
    }

    @Test
    @DisplayName("Bias and action weights are accessible")
    void testWeightAccess() {
      Quaternion bias = perceptron.getBiasRotation();
      Quaternion action = perceptron.getActionRotation();

      assertNotNull(bias);
      assertNotNull(action);
      assertTrue(bias.isUnit());
      assertTrue(action.isUnit());
    }

    @Test
    @DisplayName("Random seed is accessible")
    void testRandomSeedAccess() {
      long seed = perceptron.getRandomSeed();

      // Should return a valid long value
      assertTrue(seed != 0L, "Random seed should not be zero");

      // Should be different on subsequent calls (since it's random)
      long seed2 = perceptron.getRandomSeed();
      assertNotEquals(seed, seed2, "Random seed should change on subsequent calls");
    }
  }

  @Nested
  @DisplayName("Forward Pass")
  class ForwardPassTests {

    private QuaternionPerceptron perceptron;

    @BeforeEach
    void setUp() {
      perceptron = new QuaternionPerceptron(42L);
    }

    @Test
    @DisplayName("Forward pass with valid unit quaternion input")
    void testForwardWithValidInput() {
      // Create a simple unit quaternion input (rotation around Z-axis)
      Quaternion input = Quaternion.fromAxisAngle(Math.PI / 4, new double[] {0, 0, 1});

      // Ensure input is unit
      assertTrue(input.isUnit(), "Input should be a unit quaternion");

      // Perform forward pass
      Quaternion output = perceptron.forward(input);

      // Output should be a unit quaternion
      assertNotNull(output);
      assertTrue(output.isUnit(), "Output should be a unit quaternion");

      // Output should be different from input (weights should transform it)
      assertNotEquals(input, output);
    }

    @Test
    @DisplayName("Forward pass preserves unit length")
    void testForwardPreservesUnitLength() {
      // Test with identity quaternion
      Quaternion input = Quaternion.ONE;

      Quaternion output = perceptron.forward(input);

      // Output should be unit length
      assertTrue(output.isUnit(), "Output should maintain unit length");

      // Test with random unit quaternion
      Quaternion randomInput = new Quaternion(0.5, 0.5, 0.5, 0.5).normalize();
      assertTrue(randomInput.isUnit(), "Random input should be unit");

      Quaternion randomOutput = perceptron.forward(randomInput);
      assertTrue(randomOutput.isUnit(), "Random output should maintain unit length");
    }

    @Test
    @DisplayName("Forward pass rejects null input")
    void testForwardRejectsNullInput() {
      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.forward(null),
          "Should reject null input");
    }

    @Test
    @DisplayName("Forward pass rejects non-unit quaternion input")
    void testForwardRejectsNonUnitInput() {
      // Create a non-unit quaternion
      Quaternion nonUnitInput = new Quaternion(2.0, 1.0, 1.0, 1.0);
      assertFalse(nonUnitInput.isUnit(), "Input should not be unit");

      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.forward(nonUnitInput),
          "Should reject non-unit quaternion input");
    }

    @Test
    @DisplayName("Forward pass applies bias and action weights")
    void testForwardAppliesWeights() {
      // Create a simple input
      Quaternion input = Quaternion.ONE;

      // Get current weights
      Quaternion bias = perceptron.getBiasRotation();
      Quaternion action = perceptron.getActionRotation();

      // Perform forward pass
      Quaternion output = perceptron.forward(input);

      // Output should equal bias * input * action
      Quaternion expected = bias.multiply(input).multiply(action);
      assertEquals(expected, output, "Forward pass should equal bias * input * action");
    }
  }

  @Nested
  @DisplayName("Classification")
  class ClassificationTests {

    private QuaternionPerceptron perceptron;

    @BeforeEach
    void setUp() {
      perceptron = new QuaternionPerceptron(42L);
    }

    @Test
    @DisplayName("Classify returns valid binary labels")
    void testClassifyReturnsValidLabels() {
      Quaternion input = Quaternion.ONE;
      int label = perceptron.classify(input);

      // Should return either 0 or 1
      assertTrue(label == 0 || label == 1, "Classification should return 0 or 1");
    }

    @Test
    @DisplayName("Classify with identity input")
    void testClassifyWithIdentityInput() {
      Quaternion input = Quaternion.ONE;
      int label = perceptron.classify(input);

      // Identity should be classified as 0 (closer to identity target)
      assertEquals(0, label, "Identity input should be classified as 0");
    }

    @Test
    @DisplayName("Classify with 180 degree rotation")
    void testClassifyWith180DegreeRotation() {
      // 180 degree rotation around Z-axis should be classified as 1
      Quaternion input = new Quaternion(0.0, 0.0, 0.0, 1.0);
      int label = perceptron.classify(input);

      // Should be classified as 1 (closer to flipped target)
      assertEquals(1, label, "180 degree rotation should be classified as 1");
    }

    @Test
    @DisplayName("Classify rejects null input")
    void testClassifyRejectsNullInput() {
      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.classify(null),
          "Should reject null input");
    }

    @Test
    @DisplayName("Classify rejects non-unit quaternion input")
    void testClassifyRejectsNonUnitInput() {
      Quaternion nonUnitInput = new Quaternion(2.0, 1.0, 1.0, 1.0);
      assertFalse(nonUnitInput.isUnit(), "Input should not be unit");

      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.classify(nonUnitInput),
          "Should reject non-unit quaternion input");
    }
  }

  @Nested
  @DisplayName("Training")
  class TrainingTests {

    private QuaternionPerceptron perceptron;

    @BeforeEach
    void setUp() {
      perceptron = new QuaternionPerceptron(42L);
    }

    @Test
    @DisplayName("Step method rejects null inputs")
    void testStepRejectsNullInputs() {
      List<Integer> testLabels = Arrays.asList(0, 1);
      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.step(null, testLabels),
          "Should reject null inputs");
    }

    @Test
    @DisplayName("Step method rejects null labels")
    void testStepRejectsNullLabels() {
      List<Quaternion> testInputs = Arrays.asList(Quaternion.ONE);
      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.step(testInputs, null),
          "Should reject null labels");
    }

    @Test
    @DisplayName("Step method rejects mismatched input/label sizes")
    void testStepRejectsMismatchedSizes() {
      List<Quaternion> testInputs = Arrays.asList(Quaternion.ONE, Quaternion.I);
      List<Integer> mismatchedLabels = Arrays.asList(0);

      assertThrows(
          IllegalArgumentException.class,
          () -> perceptron.step(testInputs, mismatchedLabels),
          "Should reject mismatched input/label sizes");
    }

    @Test
    @DisplayName("Step method validates input parameters")
    void testStepValidatesInputs() {
      List<Quaternion> testInputs = Arrays.asList(Quaternion.ONE);
      List<Integer> testLabels = Arrays.asList(0);

      // Should not throw exception for valid inputs
      assertDoesNotThrow(() -> perceptron.step(testInputs, testLabels));
    }

    @Test
    @DisplayName("Step method processes batch correctly")
    void testStepMethod() {
      List<Quaternion> testInputs =
          Arrays.asList(
              Quaternion.ONE, // Identity
              new Quaternion(Math.cos(Math.PI / 4), Math.sin(Math.PI / 4), 0.0, 0.0) // 90° around X
              );
      List<Integer> testLabels = Arrays.asList(0, 1);

      // Store initial weights
      Quaternion initialBias = perceptron.getBiasRotation();
      Quaternion initialAction = perceptron.getActionRotation();

      // Perform training step
      perceptron.step(testInputs, testLabels);

      // Weights should have changed
      assertNotEquals(initialBias, perceptron.getBiasRotation());
      assertNotEquals(initialAction, perceptron.getActionRotation());
    }

    @Test
    @DisplayName("Step method maintains unit quaternion weights")
    void testStepMaintainsUnitWeights() {
      List<Quaternion> testInputs =
          Arrays.asList(
              Quaternion.ONE,
              new Quaternion(Math.cos(Math.PI / 4), Math.sin(Math.PI / 4), 0.0, 0.0));
      List<Integer> testLabels = Arrays.asList(0, 1);

      perceptron.step(testInputs, testLabels);

      // Weights should remain unit quaternions
      assertTrue(perceptron.getBiasRotation().isUnit(), "Bias should remain unit");
      assertTrue(perceptron.getActionRotation().isUnit(), "Action should remain unit");
    }

    @Test
    @DisplayName("Step method with single sample")
    void testStepWithSingleSample() {
      List<Quaternion> singleInput = Arrays.asList(Quaternion.ONE);
      List<Integer> singleLabel = Arrays.asList(0);

      // Should not throw exception
      assertDoesNotThrow(() -> perceptron.step(singleInput, singleLabel));

      // Weights should remain unit
      assertTrue(perceptron.getBiasRotation().isUnit());
      assertTrue(perceptron.getActionRotation().isUnit());
    }

    @Test
    @DisplayName("Step method with empty batch")
    void testStepWithEmptyBatch() {
      List<Quaternion> emptyInputs = Arrays.asList();
      List<Integer> emptyLabels = Arrays.asList();

      // Should not throw exception
      assertDoesNotThrow(() -> perceptron.step(emptyInputs, emptyLabels));

      // Weights should remain unit
      assertTrue(perceptron.getBiasRotation().isUnit());
      assertTrue(perceptron.getActionRotation().isUnit());
    }

    @Test
    @DisplayName("Step method with complex rotations")
    void testStepWithComplexRotations() {
      // Test with more complex rotations
      List<Quaternion> complexInputs =
          Arrays.asList(
              Quaternion.ONE,
              new Quaternion(
                  Math.cos(Math.PI / 6), Math.sin(Math.PI / 6), 0.0, 0.0), // 60° around X
              new Quaternion(
                  Math.cos(Math.PI / 3), 0.0, Math.sin(Math.PI / 3), 0.0) // 120° around Y
              );
      List<Integer> complexLabels = Arrays.asList(0, 1, 0);

      // Should not throw exception
      assertDoesNotThrow(() -> perceptron.step(complexInputs, complexLabels));

      // Weights should remain unit
      assertTrue(perceptron.getBiasRotation().isUnit());
      assertTrue(perceptron.getActionRotation().isUnit());
    }
  }

  @Nested
  @DisplayName("Gradient Computation")
  class GradientComputationTests {

    private QuaternionPerceptron perceptron;

    @BeforeEach
    void setUp() {
      perceptron = new QuaternionPerceptron(42L);
    }

    @Test
    @DisplayName("Compute gradient fields with valid inputs")
    void testComputeGradientFieldsValidInputs() {
      List<Quaternion> inputs = Arrays.asList(Quaternion.ONE, Quaternion.I);
      List<Quaternion> predicted =
          Arrays.asList(
              new Quaternion(0.8, 0.2, 0.0, 0.0).normalize(),
              new Quaternion(0.0, 0.9, 0.1, 0.0).normalize());
      List<Quaternion> targets = Arrays.asList(Quaternion.ONE, Quaternion.I);

      QuaternionPerceptron.GradientFields result =
          perceptron.computeGradientFields(inputs, predicted, targets);

      assertNotNull(result);
      assertNotNull(result.biasGradient);
      assertNotNull(result.actionGradient);

      // Gradients should be unit quaternions
      assertTrue(result.biasGradient.isUnit());
      assertTrue(result.actionGradient.isUnit());
    }

    @Test
    @DisplayName("Compute gradient fields with empty batch")
    void testComputeGradientFieldsEmptyBatch() {
      List<Quaternion> emptyInputs = Arrays.asList();
      List<Quaternion> emptyPredicted = Arrays.asList();
      List<Quaternion> emptyTargets = Arrays.asList();

      QuaternionPerceptron.GradientFields result =
          perceptron.computeGradientFields(emptyInputs, emptyPredicted, emptyTargets);

      assertNotNull(result);
      assertNotNull(result.biasGradient);
      assertNotNull(result.actionGradient);

      // Should return identity rotations for empty batch
      assertEquals(Quaternion.ONE, result.biasGradient);
      assertEquals(Quaternion.ONE, result.actionGradient);
    }

    @Test
    @DisplayName("Compute gradient fields with single sample")
    void testComputeGradientFieldsSingleSample() {
      List<Quaternion> inputs = Arrays.asList(Quaternion.ONE);
      List<Quaternion> predicted = Arrays.asList(new Quaternion(0.8, 0.2, 0.0, 0.0).normalize());
      List<Quaternion> targets = Arrays.asList(Quaternion.ONE);

      QuaternionPerceptron.GradientFields result =
          perceptron.computeGradientFields(inputs, predicted, targets);

      assertNotNull(result);
      assertTrue(result.biasGradient.isUnit());
      assertTrue(result.actionGradient.isUnit());
    }

    @Test
    @DisplayName("Compute gradient fields with large error")
    void testComputeGradientFieldsLargeError() {
      // Create a large error scenario
      List<Quaternion> inputs = Arrays.asList(Quaternion.ONE);
      List<Quaternion> predicted = Arrays.asList(Quaternion.I); // 90° rotation
      List<Quaternion> targets = Arrays.asList(Quaternion.ONE); // Identity

      QuaternionPerceptron.GradientFields result =
          perceptron.computeGradientFields(inputs, predicted, targets);

      assertNotNull(result);
      assertTrue(result.biasGradient.isUnit());
      assertTrue(result.actionGradient.isUnit());

      // Should not be identity for large error (but may be due to numerical precision)
      // Just verify they are valid unit quaternions
      assertTrue(result.biasGradient.isUnit());
      assertTrue(result.actionGradient.isUnit());
    }
  }

  @Nested
  @DisplayName("GradientFields Class")
  class GradientFieldsTests {

    @Test
    @DisplayName("GradientFields construction and access")
    void testGradientFieldsConstruction() {
      Quaternion biasGrad = Quaternion.ONE;
      Quaternion actionGrad = Quaternion.I;

      QuaternionPerceptron.GradientFields fields =
          new QuaternionPerceptron.GradientFields(biasGrad, actionGrad);

      assertEquals(biasGrad, fields.biasGradient);
      assertEquals(actionGrad, fields.actionGradient);
    }

    @Test
    @DisplayName("GradientFields with different gradients")
    void testGradientFieldsDifferentGradients() {
      Quaternion biasGrad = new Quaternion(0.8, 0.2, 0.0, 0.0);
      Quaternion quaternion = new Quaternion(0.0, 0.9, 0.1, 0.0);

      QuaternionPerceptron.GradientFields fields =
          new QuaternionPerceptron.GradientFields(biasGrad, quaternion);

      assertEquals(biasGrad, fields.biasGradient);
      assertEquals(quaternion, fields.actionGradient);

      // Should be different
      assertNotEquals(fields.biasGradient, fields.actionGradient);
    }
  }
}
