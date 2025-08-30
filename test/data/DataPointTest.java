package data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class DataPointTest {

  private double[] validCoordinates;
  private DataPoint dataPoint;

  @BeforeEach
  void setUp() {
    validCoordinates = new double[] {1.0, 2.0, 3.0};
    dataPoint = new DataPoint(validCoordinates, 1);
  }

  @Test
  void testValidConstruction() {
    assertNotNull(dataPoint);
    assertEquals(1, dataPoint.label());
    assertArrayEquals(validCoordinates, dataPoint.coordinates());
    assertEquals(3, dataPoint.getNumDimensions());
  }

  @Test
  void testNullCoordinatesThrowsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          new DataPoint(null, 0);
        });
  }

  @Test
  void testEmptyCoordinatesThrowsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          new DataPoint(new double[] {}, 0);
        });
  }

  @Test
  void testGetCoordinateValidIndex() {
    assertEquals(1.0, dataPoint.getCoordinate(0));
    assertEquals(2.0, dataPoint.getCoordinate(1));
    assertEquals(3.0, dataPoint.getCoordinate(2));
  }

  @Test
  void testGetCoordinateNegativeIndexThrowsException() {
    assertThrows(
        IndexOutOfBoundsException.class,
        () -> {
          dataPoint.getCoordinate(-1);
        });
  }

  @Test
  void testGetCoordinateOutOfBoundsIndexThrowsException() {
    assertThrows(
        IndexOutOfBoundsException.class,
        () -> {
          dataPoint.getCoordinate(3);
        });
  }

  @Test
  void testGetNumDimensions() {
    assertEquals(3, dataPoint.getNumDimensions());

    DataPoint singleDim = new DataPoint(new double[] {5.0}, 0);
    assertEquals(1, singleDim.getNumDimensions());

    DataPoint twoDim = new DataPoint(new double[] {1.0, 2.0}, 1);
    assertEquals(2, twoDim.getNumDimensions());
  }

  @Test
  void testRecordImmutability() {
    double[] originalCoords = dataPoint.coordinates();
    originalCoords[0] = 999.0;
    assertEquals(999.0, dataPoint.getCoordinate(0));
  }

  @Test
  void testEqualsAndHashCode() {
    DataPoint samePoint = new DataPoint(validCoordinates, 1);
    DataPoint differentLabel = new DataPoint(validCoordinates, 0);
    DataPoint differentCoords = new DataPoint(new double[] {1.0, 2.0, 4.0}, 1);

    assertEquals(dataPoint, samePoint);
    assertNotEquals(dataPoint, differentLabel);
    assertNotEquals(dataPoint, differentCoords);

    assertEquals(dataPoint.hashCode(), samePoint.hashCode());
  }

  @Test
  void testToString() {
    String str = dataPoint.toString();
    assertNotNull(str);
    assertFalse(str.isEmpty());
  }
}
