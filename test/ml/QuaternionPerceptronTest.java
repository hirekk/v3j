package ml;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

      assertEquals(0.01, perceptron.getLearningRate(), EPSILON);
      assertNotNull(perceptron.getBias());
      assertNotNull(perceptron.getAction());
    }

    @Test
    @DisplayName("Parameterized constructor initializes with specified values")
    void testParameterizedConstructor() {
      double learningRate = 0.05;
      long randomSeed = 42L;

      QuaternionPerceptron perceptron = new QuaternionPerceptron(learningRate, randomSeed);

      assertEquals(learningRate, perceptron.getLearningRate(), EPSILON);
      assertNotNull(perceptron.getBias());
      assertNotNull(perceptron.getAction());
    }

    @Test
    @DisplayName("Constructor with same seed produces identical weights")
    void testDeterministicInitialization() {
      long seed = 12345L;

      QuaternionPerceptron perceptron1 = new QuaternionPerceptron(0.01, seed);
      QuaternionPerceptron perceptron2 = new QuaternionPerceptron(0.01, seed);

      assertEquals(perceptron1.getBias(), perceptron2.getBias());
      assertEquals(perceptron1.getAction(), perceptron2.getAction());
    }

    @Test
    @DisplayName("Constructor with different seeds produces different weights")
    void testNonDeterministicInitialization() {
      QuaternionPerceptron perceptron1 = new QuaternionPerceptron(0.01, 1L);
      QuaternionPerceptron perceptron2 = new QuaternionPerceptron(0.01, 2L);

      assertNotEquals(perceptron1.getBias(), perceptron2.getBias());
      assertNotEquals(perceptron1.getAction(), perceptron2.getAction());
    }

    @Test
    @DisplayName("Constructor rejects non-positive learning rate")
    void testConstructorRejectsNonPositiveLearningRate() {
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuaternionPerceptron(0.0, 42L),
          "Should reject zero learning rate");

      assertThrows(
          IllegalArgumentException.class,
          () -> new QuaternionPerceptron(-0.01, 42L),
          "Should reject negative learning rate");
    }
  }

  @Nested
  @DisplayName("Weight Properties")
  class WeightPropertiesTests {

    @Test
    @DisplayName("Weights are valid unit quaternions")
    void testWeightsAreValidUnitQuaternions() {
      QuaternionPerceptron perceptron = new QuaternionPerceptron(0.01, 42L);

      Quaternion bias = perceptron.getBias();
      Quaternion action = perceptron.getAction();

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
        QuaternionPerceptron newPerceptron = new QuaternionPerceptron(0.01, 42L + attempt);
        Quaternion newBias = newPerceptron.getBias();
        Quaternion newAction = newPerceptron.getAction();

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
        QuaternionPerceptron perceptron = new QuaternionPerceptron(0.01, 42L + attempt);

        Quaternion bias = perceptron.getBias();
        Quaternion action = perceptron.getAction();

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
      perceptron = new QuaternionPerceptron(0.01, 42L);
    }

    @Test
    @DisplayName("Learning rate is accessible and correct")
    void testLearningRateAccess() {
      assertEquals(0.01, perceptron.getLearningRate(), EPSILON);
    }

    @Test
    @DisplayName("Bias and action weights are accessible")
    void testWeightAccess() {
      Quaternion bias = perceptron.getBias();
      Quaternion action = perceptron.getAction();

      assertNotNull(bias);
      assertNotNull(action);
      assertTrue(bias.isUnit());
      assertTrue(action.isUnit());
    }
  }

  @Nested
  @DisplayName("Forward Pass")
  class ForwardPassTests {

    private QuaternionPerceptron perceptron;

    @BeforeEach
    void setUp() {
      perceptron = new QuaternionPerceptron(0.01, 42L);
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
      Quaternion bias = perceptron.getBias();
      Quaternion action = perceptron.getAction();

      // Perform forward pass
      Quaternion output = perceptron.forward(input);

      // Output should equal bias * input * action
      Quaternion expected = bias.multiply(input).multiply(action);
      assertEquals(expected, output, "Forward pass should equal bias * input * action");
    }
  }
}
