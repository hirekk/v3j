package math;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("Quaternion Tests")
class QuaternionTest {

  private Quaternion q1, q2, q3;
  private static final double EPSILON = 1e-10;

  @BeforeEach
  void setUp() {
    q1 = new Quaternion(1.0, 2.0, 3.0, 4.0);
    q2 = new Quaternion(2.0, 1.0, 4.0, 3.0);
    q3 = new Quaternion(0.0, 1.0, 0.0, 0.0);
  }

  @Nested
  @DisplayName("Construction Tests")
  class ConstructionTests {

    @Test
    @DisplayName("Basic construction with all components")
    void testConstruction() {
      Quaternion q = new Quaternion(1.0, 2.0, 3.0, 4.0);
      assertEquals(1.0, q.getW(), EPSILON);
      assertEquals(2.0, q.getX(), EPSILON);
      assertEquals(3.0, q.getY(), EPSILON);
      assertEquals(4.0, q.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Construction from vector array")
    void testConstructionFromVector() {
      double[] vector = {2.0, 3.0, 4.0};
      Quaternion q = new Quaternion(1.0, vector);
      assertEquals(1.0, q.getW(), EPSILON);
      assertArrayEquals(vector, q.getVector(), EPSILON);
    }

    @Test
    @DisplayName("Construction with invalid vector length")
    void testConstructionFromVectorInvalidLength() {
      double[] invalidVector = {1.0, 2.0};
      assertThrows(IllegalArgumentException.class, () -> new Quaternion(1.0, invalidVector));
    }

    @Test
    @DisplayName("Construction with null vector")
    void testConstructionFromNullVector() {
      assertThrows(NullPointerException.class, () -> new Quaternion(1.0, (double[]) null));
    }

    @Test
    @DisplayName("Construction with empty vector")
    void testConstructionFromEmptyVector() {
      double[] emptyVector = {};
      assertThrows(IllegalArgumentException.class, () -> new Quaternion(1.0, emptyVector));
    }

    @Test
    @DisplayName("Construction with floating point components")
    void testConstructionWithFloatingPoint() {
      Quaternion q = new Quaternion(1.5, -2.7, 3.14159, -4.2);
      assertEquals(1.5, q.getW(), EPSILON);
      assertEquals(-2.7, q.getX(), EPSILON);
      assertEquals(3.14159, q.getY(), EPSILON);
      assertEquals(-4.2, q.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Constant Tests")
  class ConstantTests {

    @Test
    @DisplayName("Zero quaternion constant")
    void testZeroConstant() {
      assertEquals(0.0, Quaternion.ZERO.getW(), EPSILON);
      assertEquals(0.0, Quaternion.ZERO.getX(), EPSILON);
      assertEquals(0.0, Quaternion.ZERO.getY(), EPSILON);
      assertEquals(0.0, Quaternion.ZERO.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Unit quaternion constant")
    void testOneConstant() {
      assertEquals(1.0, Quaternion.ONE.getW(), EPSILON);
      assertEquals(0.0, Quaternion.ONE.getX(), EPSILON);
      assertEquals(0.0, Quaternion.ONE.getY(), EPSILON);
      assertEquals(0.0, Quaternion.ONE.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Unit vector constants")
    void testUnitVectorConstants() {
      assertEquals(1.0, Quaternion.I.getX(), EPSILON);
      assertEquals(1.0, Quaternion.J.getY(), EPSILON);
      assertEquals(1.0, Quaternion.K.getZ(), EPSILON);

      // Verify other components are zero
      assertEquals(0.0, Quaternion.I.getW(), EPSILON);
      assertEquals(0.0, Quaternion.I.getY(), EPSILON);
      assertEquals(0.0, Quaternion.I.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Getter Tests")
  class GetterTests {

    @Test
    @DisplayName("Scalar and vector getters")
    void testGetters() {
      assertEquals(1.0, q1.getScalar(), EPSILON);
      assertArrayEquals(new double[] {2.0, 3.0, 4.0}, q1.getVector(), EPSILON);
    }

    @Test
    @DisplayName("Individual component getters")
    void testComponentGetters() {
      assertEquals(1.0, q1.getW(), EPSILON);
      assertEquals(2.0, q1.getX(), EPSILON);
      assertEquals(3.0, q1.getY(), EPSILON);
      assertEquals(4.0, q1.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Property Tests")
  class PropertyTests {

    @Test
    @DisplayName("Scalar quaternion detection")
    void testIsScalar() {
      assertTrue(Quaternion.ONE.isScalar());
      assertFalse(q1.isScalar());
      assertFalse(Quaternion.I.isScalar());
    }

    @Test
    @DisplayName("Vector quaternion detection")
    void testIsVector() {
      assertTrue(q3.isVector());
      assertFalse(q1.isVector());
      assertFalse(Quaternion.ONE.isVector());
    }

    @Test
    @DisplayName("Unit quaternion detection")
    void testIsUnit() {
      Quaternion unit = new Quaternion(1.0, 0.0, 0.0, 0.0);
      assertTrue(unit.isUnit());
      assertFalse(q1.isUnit());

      // Test normalized quaternion
      Quaternion normalized = q1.normalize();
      assertTrue(normalized.isUnit());
    }

    @Test
    @DisplayName("Zero quaternion detection")
    void testIsZero() {
      assertTrue(Quaternion.ZERO.isZero());
      assertFalse(q1.isZero());
      assertFalse(Quaternion.ONE.isZero());
    }

    @Test
    @DisplayName("Near-zero quaternion detection")
    void testIsZeroNearZero() {
      Quaternion nearZero = new Quaternion(1e-15, 1e-15, 1e-15, 1e-15);
      assertFalse(nearZero.isZero());

      Quaternion actualZero = new Quaternion(0.0, 0.0, 0.0, 0.0);
      assertTrue(actualZero.isZero());
    }
  }

  @Nested
  @DisplayName("Arithmetic Operation Tests")
  class ArithmeticTests {

    @Test
    @DisplayName("Quaternion addition")
    void testAdd() {
      Quaternion result = q1.add(q2);
      assertEquals(3.0, result.getW(), EPSILON);
      assertEquals(3.0, result.getX(), EPSILON);
      assertEquals(7.0, result.getY(), EPSILON);
      assertEquals(7.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Addition with zero quaternion")
    void testAddWithZero() {
      Quaternion result = q1.add(Quaternion.ZERO);
      assertEquals(q1.getW(), result.getW(), EPSILON);
      assertEquals(q1.getX(), result.getX(), EPSILON);
      assertEquals(q1.getY(), result.getY(), EPSILON);
      assertEquals(q1.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Quaternion subtraction")
    void testSubtract() {
      Quaternion result = q1.subtract(q2);
      assertEquals(-1.0, result.getW(), EPSILON);
      assertEquals(1.0, result.getX(), EPSILON);
      assertEquals(-1.0, result.getY(), EPSILON);
      assertEquals(1.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Subtraction from zero")
    void testSubtractFromZero() {
      Quaternion result = Quaternion.ZERO.subtract(q1);
      assertEquals(-q1.getW(), result.getW(), EPSILON);
      assertEquals(-q1.getX(), result.getX(), EPSILON);
      assertEquals(-q1.getY(), result.getY(), EPSILON);
      assertEquals(-q1.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Quaternion multiplication")
    void testMultiplyQuaternion() {
      Quaternion result = q1.multiply(q2);
      assertEquals(-24.0, result.getW(), EPSILON);
      assertEquals(-2.0, result.getX(), EPSILON);
      assertEquals(8.0, result.getY(), EPSILON);
      assertEquals(16.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Multiplication by identity")
    void testMultiplyByIdentity() {
      Quaternion result = q1.multiply(Quaternion.ONE);
      assertEquals(q1.getW(), result.getW(), EPSILON);
      assertEquals(q1.getX(), result.getX(), EPSILON);
      assertEquals(q1.getY(), result.getY(), EPSILON);
      assertEquals(q1.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Multiplication by zero")
    void testMultiplyByZero() {
      Quaternion result = q1.multiply(Quaternion.ZERO);
      assertEquals(Quaternion.ZERO.getW(), result.getW(), EPSILON);
      assertEquals(Quaternion.ZERO.getX(), result.getX(), EPSILON);
      assertEquals(Quaternion.ZERO.getY(), result.getY(), EPSILON);
      assertEquals(Quaternion.ZERO.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Scalar multiplication")
    void testMultiplyScalar() {
      Quaternion result = q1.multiply(2.0);
      assertEquals(2.0, result.getW(), EPSILON);
      assertEquals(4.0, result.getX(), EPSILON);
      assertEquals(6.0, result.getY(), EPSILON);
      assertEquals(8.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Multiplication by negative scalar")
    void testMultiplyByNegativeScalar() {
      Quaternion result = q1.multiply(-1.0);
      assertEquals(-q1.getW(), result.getW(), EPSILON);
      assertEquals(-q1.getX(), result.getX(), EPSILON);
      assertEquals(-q1.getY(), result.getY(), EPSILON);
      assertEquals(-q1.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Division by scalar")
    void testDivide() {
      Quaternion result = q1.divide(2.0);
      assertEquals(0.5, result.getW(), EPSILON);
      assertEquals(1.0, result.getX(), EPSILON);
      assertEquals(1.5, result.getY(), EPSILON);
      assertEquals(2.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Division by zero")
    void testDivideByZero() {
      assertThrows(ArithmeticException.class, () -> q1.divide(0.0));
    }

    @Test
    @DisplayName("Division by very small number")
    void testDivideByVerySmallNumber() {
      Quaternion result = q1.divide(1e-10);
      assertEquals(q1.getW() * 1e10, result.getW(), EPSILON);
      assertEquals(q1.getX() * 1e10, result.getX(), EPSILON);
      assertEquals(q1.getY() * 1e10, result.getY(), EPSILON);
      assertEquals(q1.getZ() * 1e10, result.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Unary Operation Tests")
  class UnaryOperationTests {

    @Test
    @DisplayName("Quaternion negation")
    void testNegate() {
      Quaternion result = q1.negate();
      assertEquals(-1.0, result.getW(), EPSILON);
      assertEquals(-2.0, result.getX(), EPSILON);
      assertEquals(-3.0, result.getY(), EPSILON);
      assertEquals(-4.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Negation of zero")
    void testNegateZero() {
      Quaternion result = Quaternion.ZERO.negate();
      assertEquals(Quaternion.ZERO.getW(), result.getW(), EPSILON);
      assertEquals(Quaternion.ZERO.getX(), result.getX(), EPSILON);
      assertEquals(Quaternion.ZERO.getY(), result.getY(), EPSILON);
      assertEquals(Quaternion.ZERO.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Quaternion conjugation")
    void testConjugate() {
      Quaternion result = q1.conjugate();
      assertEquals(1.0, result.getW(), EPSILON);
      assertEquals(-2.0, result.getX(), EPSILON);
      assertEquals(-3.0, result.getY(), EPSILON);
      assertEquals(-4.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Conjugation of scalar quaternion")
    void testConjugateScalar() {
      Quaternion result = Quaternion.ONE.conjugate();
      assertEquals(1.0, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Norm and Normalization Tests")
  class NormTests {

    @Test
    @DisplayName("Quaternion norm")
    void testNorm() {
      double expected = Math.sqrt(1.0 + 4.0 + 9.0 + 16.0);
      assertEquals(expected, q1.norm(), EPSILON);
    }

    @Test
    @DisplayName("Zero quaternion norm")
    void testZeroNorm() {
      assertEquals(0.0, Quaternion.ZERO.norm(), EPSILON);
    }

    @Test
    @DisplayName("Unit quaternion norm")
    void testUnitNorm() {
      assertEquals(1.0, Quaternion.ONE.norm(), EPSILON);
    }

    @Test
    @DisplayName("Squared norm")
    void testNormSquared() {
      double expected = 1.0 + 4.0 + 9.0 + 16.0;
      assertEquals(expected, q1.normSquared(), EPSILON);
    }

    @Test
    @DisplayName("Normalization")
    void testNormalize() {
      Quaternion normalized = q1.normalize();
      assertEquals(1.0, normalized.norm(), EPSILON);

      // Verify direction is preserved
      double scale = q1.norm();
      assertEquals(q1.getW() / scale, normalized.getW(), EPSILON);
      assertEquals(q1.getX() / scale, normalized.getX(), EPSILON);
      assertEquals(q1.getY() / scale, normalized.getY(), EPSILON);
      assertEquals(q1.getZ() / scale, normalized.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Normalize zero quaternion")
    void testNormalizeZero() {
      assertThrows(ArithmeticException.class, () -> Quaternion.ZERO.normalize());
    }

    @Test
    @DisplayName("Normalize very small quaternion")
    void testNormalizeVerySmall() {
      Quaternion verySmall = new Quaternion(1e-15, 1e-15, 1e-15, 1e-15);
      Quaternion normalized = verySmall.normalize();
      assertEquals(1.0, normalized.norm(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Inverse Tests")
  class InverseTests {

    @Test
    @DisplayName("Quaternion inverse")
    void testInverse() {
      Quaternion inverse = q1.inverse();
      Quaternion product = q1.multiply(inverse);

      // Should get identity quaternion (approximately)
      assertEquals(1.0, product.getW(), EPSILON);
      assertEquals(0.0, product.getX(), EPSILON);
      assertEquals(0.0, product.getY(), EPSILON);
      assertEquals(0.0, product.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Inverse of zero quaternion")
    void testInverseZero() {
      assertThrows(ArithmeticException.class, () -> Quaternion.ZERO.inverse());
    }

    @Test
    @DisplayName("Inverse of unit quaternion")
    void testInverseUnit() {
      Quaternion inverse = Quaternion.ONE.inverse();
      assertEquals(Quaternion.ONE.getW(), inverse.getW(), EPSILON);
      assertEquals(Quaternion.ONE.getX(), inverse.getX(), EPSILON);
      assertEquals(Quaternion.ONE.getY(), inverse.getY(), EPSILON);
      assertEquals(Quaternion.ONE.getZ(), inverse.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Product Tests")
  class ProductTests {

    @Test
    @DisplayName("Dot product")
    void testDot() {
      double dot = q1.dot(q2);
      double expected = 1.0 * 2.0 + 2.0 * 1.0 + 3.0 * 4.0 + 4.0 * 3.0;
      assertEquals(expected, dot, EPSILON);
    }

    @Test
    @DisplayName("Dot product with zero")
    void testDotWithZero() {
      assertEquals(0.0, q1.dot(Quaternion.ZERO), EPSILON);
    }

    @Test
    @DisplayName("Cross product")
    void testCross() {
      Quaternion result = q3.cross(Quaternion.J);
      assertEquals(0.0, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(1.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Cross product with non-vector quaternion")
    void testCrossNonVector() {
      assertThrows(IllegalArgumentException.class, () -> q1.cross(q2));
    }

    @Test
    @DisplayName("Cross product with zero vector")
    void testCrossWithZero() {
      Quaternion zeroVector = new Quaternion(0.0, 0.0, 0.0, 0.0);
      Quaternion result = q3.cross(zeroVector);
      assertEquals(0.0, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Exponential and Logarithm Tests")
  class ExponentialTests {

    @Test
    @DisplayName("Exponential of vector quaternion")
    void testExp() {
      Quaternion result = q3.exp();
      assertEquals(Math.cos(1.0), result.getW(), EPSILON);
      assertEquals(Math.sin(1.0), result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Exponential of scalar quaternion")
    void testExpScalar() {
      Quaternion result = Quaternion.ONE.exp();
      assertEquals(Math.E, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Exponential of zero quaternion")
    void testExpZero() {
      Quaternion result = Quaternion.ZERO.exp();
      assertEquals(1.0, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Logarithm of unit quaternion")
    void testLog() {
      Quaternion result = Quaternion.ONE.log();
      assertEquals(0.0, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Logarithm of zero quaternion")
    void testLogZero() {
      assertThrows(ArithmeticException.class, () -> Quaternion.ZERO.log());
    }

    @Test
    @DisplayName("Logarithm of vector quaternion")
    void testLogVector() {
      Quaternion result = q3.log();
      assertEquals(0.0, result.getW(), EPSILON);
      assertEquals(Math.PI / 2, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Power Tests")
  class PowerTests {

    @Test
    @DisplayName("Power with positive exponent")
    void testPow() {
      Quaternion result = q1.pow(2.0);
      Quaternion expected = q1.multiply(q1);
      assertEquals(expected.getW(), result.getW(), EPSILON);
      assertEquals(expected.getX(), result.getX(), EPSILON);
      assertEquals(expected.getY(), result.getY(), EPSILON);
      assertEquals(expected.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Power special cases")
    void testPowSpecialCases() {
      assertEquals(Quaternion.ONE, q1.pow(0.0));
      assertEquals(q1, q1.pow(1.0));
      assertEquals(q1.inverse(), q1.pow(-1.0));
    }

    @Test
    @DisplayName("Power with quaternion exponent")
    void testPowQuaternion() {
      Quaternion result = q1.pow(Quaternion.ONE);
      assertEquals(q1.getW(), result.getW(), EPSILON);
      assertEquals(q1.getX(), result.getX(), EPSILON);
      assertEquals(q1.getY(), result.getY(), EPSILON);
      assertEquals(q1.getZ(), result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Power with non-scalar quaternion exponent")
    void testPowQuaternionNonScalar() {
      assertThrows(UnsupportedOperationException.class, () -> q1.pow(q3));
    }

    @Test
    @DisplayName("Power with fractional exponent")
    void testPowFractional() {
      Quaternion result = q1.pow(0.5);
      Quaternion squared = result.multiply(result);
      assertEquals(q1.getW(), squared.getW(), EPSILON);
      assertEquals(q1.getX(), squared.getX(), EPSILON);
      assertEquals(q1.getY(), squared.getY(), EPSILON);
      assertEquals(q1.getZ(), squared.getZ(), EPSILON);
    }
  }

  @Nested
  @DisplayName("Rotation Matrix Tests")
  class RotationMatrixTests {

    @Test
    @DisplayName("Identity rotation matrix")
    void testToRotationMatrix() {
      Quaternion unit = new Quaternion(1.0, 0.0, 0.0, 0.0);
      double[] matrix = unit.toRotationMatrix();
      assertEquals(9, matrix.length);

      // Should be identity matrix
      assertEquals(1.0, matrix[0], EPSILON); // m11
      assertEquals(0.0, matrix[1], EPSILON); // m12
      assertEquals(0.0, matrix[2], EPSILON); // m13
      assertEquals(0.0, matrix[3], EPSILON); // m21
      assertEquals(1.0, matrix[4], EPSILON); // m22
      assertEquals(0.0, matrix[5], EPSILON); // m23
      assertEquals(0.0, matrix[6], EPSILON); // m31
      assertEquals(0.0, matrix[7], EPSILON); // m32
      assertEquals(1.0, matrix[8], EPSILON); // m33
    }

    @Test
    @DisplayName("90-degree rotation around X-axis")
    void testRotationMatrix90DegreesX() {
      Quaternion rotX = new Quaternion(Math.cos(Math.PI / 4), Math.sin(Math.PI / 4), 0.0, 0.0);
      double[] matrix = rotX.toRotationMatrix();

      // Verify rotation matrix properties
      assertEquals(1.0, matrix[0], EPSILON); // m11 should be 1
      assertEquals(0.0, matrix[1], EPSILON); // m12 should be 0
      assertEquals(0.0, matrix[2], EPSILON); // m13 should be 0
      assertEquals(0.0, matrix[3], EPSILON); // m21 should be 0
      assertEquals(0.0, matrix[4], EPSILON); // m22 should be 0 (cos(90°))
      assertEquals(-1.0, matrix[5], EPSILON); // m23 should be -1 (sin(90°))
      assertEquals(0.0, matrix[6], EPSILON); // m31 should be 0
      assertEquals(1.0, matrix[7], EPSILON); // m32 should be 1 (sin(90°))
      assertEquals(0.0, matrix[8], EPSILON); // m33 should be 0 (cos(90°))
    }

    @Test
    @DisplayName("Non-unit quaternion rotation matrix")
    void testToRotationMatrixNonUnit() {
      assertThrows(IllegalStateException.class, () -> q1.toRotationMatrix());
    }

    @Test
    @DisplayName("Rotation matrix orthogonality")
    void testRotationMatrixOrthogonality() {
      Quaternion unit = new Quaternion(1.0, 0.0, 0.0, 0.0);
      double[] matrix = unit.toRotationMatrix();

      // Verify determinant is 1 (rotation matrix property)
      double det =
          matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
              - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
              + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
      assertEquals(1.0, det, EPSILON);
    }
  }

  @Nested
  @DisplayName("Construction from Rotation Tests")
  class RotationConstructionTests {

    @Test
    @DisplayName("Axis-angle construction")
    void testFromAxisAngle() {
      double[] axis = {1.0, 0.0, 0.0};
      Quaternion result = Quaternion.fromAxisAngle(Math.PI / 2, axis);
      assertEquals(Math.cos(Math.PI / 4), result.getW(), EPSILON);
      assertEquals(Math.sin(Math.PI / 4), result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Axis-angle with invalid axis length")
    void testFromAxisAngleInvalidAxis() {
      double[] invalidAxis = {1.0, 2.0};
      assertThrows(
          IllegalArgumentException.class, () -> Quaternion.fromAxisAngle(Math.PI, invalidAxis));
    }

    @Test
    @DisplayName("Axis-angle with zero axis")
    void testFromAxisAngleZeroAxis() {
      double[] zeroAxis = {0.0, 0.0, 0.0};
      assertThrows(
          IllegalArgumentException.class, () -> Quaternion.fromAxisAngle(Math.PI, zeroAxis));
    }

    @Test
    @DisplayName("Axis-angle with null axis")
    void testFromAxisAngleNullAxis() {
      assertThrows(NullPointerException.class, () -> Quaternion.fromAxisAngle(Math.PI, null));
    }

    @Test
    @DisplayName("Euler angles construction")
    void testFromEuler() {
      Quaternion result = Quaternion.fromEuler(0.0, 0.0, 0.0);
      assertEquals(1.0, result.getW(), EPSILON);
      assertEquals(0.0, result.getX(), EPSILON);
      assertEquals(0.0, result.getY(), EPSILON);
      assertEquals(0.0, result.getZ(), EPSILON);
    }

    @Test
    @DisplayName("Euler angles with 90-degree rotations")
    void testFromEuler90Degrees() {
      Quaternion result = Quaternion.fromEuler(Math.PI / 2, Math.PI / 2, Math.PI / 2);
      assertNotNull(result);
      assertTrue(Math.abs(result.norm() - 1.0) < EPSILON);
    }
  }

  @Nested
  @DisplayName("SLERP Tests")
  class SlerpTests {

    @Test
    @DisplayName("Basic SLERP interpolation")
    void testSlerp() {
      Quaternion result = Quaternion.slerp(Quaternion.ONE, Quaternion.I, 0.5);
      assertNotNull(result);
      assertEquals(1.0, result.norm(), EPSILON);
    }

    @Test
    @DisplayName("SLERP at boundary values")
    void testSlerpBoundaries() {
      Quaternion result0 = Quaternion.slerp(Quaternion.ONE, Quaternion.I, 0.0);
      Quaternion result1 = Quaternion.slerp(Quaternion.ONE, Quaternion.I, 1.0);

      assertEquals(Quaternion.ONE.getW(), result0.getW(), EPSILON);
      assertEquals(Quaternion.ONE.getX(), result0.getX(), EPSILON);
      assertEquals(Quaternion.ONE.getY(), result0.getY(), EPSILON);
      assertEquals(Quaternion.ONE.getZ(), result0.getZ(), EPSILON);

      assertEquals(Quaternion.I.getW(), result1.getW(), EPSILON);
      assertEquals(Quaternion.I.getX(), result1.getX(), EPSILON);
      assertEquals(Quaternion.I.getY(), result1.getY(), EPSILON);
      assertEquals(Quaternion.I.getZ(), result1.getZ(), EPSILON);
    }

    @Test
    @DisplayName("SLERP with antipodal quaternions")
    void testSlerpAntipodal() {
      Quaternion antipodal = Quaternion.ONE.negate();
      Quaternion result = Quaternion.slerp(Quaternion.ONE, antipodal, 0.5);
      assertNotNull(result);
      assertEquals(1.0, result.norm(), EPSILON);
    }

    @Test
    @DisplayName("SLERP with very similar quaternions")
    void testSlerpSimilar() {
      Quaternion similar = new Quaternion(0.9999, 0.01, 0.0, 0.0);
      Quaternion result = Quaternion.slerp(Quaternion.ONE, similar, 0.5);
      assertNotNull(result);
      assertEquals(1.0, result.norm(), EPSILON);
    }

    @Test
    @DisplayName("SLERP with invalid parameter")
    void testSlerpInvalidParameter() {
      assertThrows(
          IllegalArgumentException.class,
          () -> Quaternion.slerp(Quaternion.ONE, Quaternion.I, 1.5));
      assertThrows(
          IllegalArgumentException.class,
          () -> Quaternion.slerp(Quaternion.ONE, Quaternion.I, -0.5));
    }
  }

  @Nested
  @DisplayName("Object Method Tests")
  class ObjectMethodTests {

    @Test
    @DisplayName("Equality comparison")
    void testEquals() {
      Quaternion same = new Quaternion(1.0, 2.0, 3.0, 4.0);
      Quaternion different = new Quaternion(1.0, 2.0, 3.0, 5.0);

      assertEquals(q1, same);
      assertNotEquals(q1, different);
      assertNotEquals(q1, null);
      assertNotEquals(q1, "string");
    }

    @Test
    @DisplayName("Equality with floating point precision")
    void testEqualsFloatingPoint() {
      Quaternion q1 = new Quaternion(1.0, 2.0, 3.0, 4.0);
      Quaternion q2 = new Quaternion(1.0 + 1e-15, 2.0, 3.0, 4.0);
      assertNotEquals(q1, q2);
    }

    @Test
    @DisplayName("Hash code consistency")
    void testHashCode() {
      Quaternion same = new Quaternion(1.0, 2.0, 3.0, 4.0);
      assertEquals(q1.hashCode(), same.hashCode());
    }

    @Test
    @DisplayName("String representation")
    void testToString() {
      String str = q1.toString();
      assertTrue(str.contains("1.000000"));
      assertTrue(str.contains("2.000000"));
      assertTrue(str.contains("3.000000"));
      assertTrue(str.contains("4.000000"));
      assertTrue(str.startsWith("Quaternion("));
      assertTrue(str.endsWith(")"));
    }
  }
}
