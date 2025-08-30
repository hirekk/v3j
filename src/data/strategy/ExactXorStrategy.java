package data.strategy;

import data.DataPoint;
import java.util.ArrayList;
import java.util.List;

public final class ExactXorStrategy implements GenerationStrategy {
  @Override
  public List<DataPoint> generateDataset(int numDimensions) {
    int numVertices = 1 << numDimensions;
    List<DataPoint> dataset = new ArrayList<>(numVertices);

    for (int i = 0; i < numVertices; i++) {
      double[] coordinates = vertexCoordinatesFromIndex(i, numDimensions);
      int label = calculateVertexParity(coordinates);
      dataset.add(new DataPoint(coordinates, label));
    }

    return dataset;
  }

  private double[] vertexCoordinatesFromIndex(int vertexIndex, int numDimensions) {
    double[] coordinates = new double[numDimensions];
    for (int i = 0; i < numDimensions; i++) {
      coordinates[i] = ((vertexIndex >> i) & 1) == 1 ? 1.0 : 0.0;
    }
    return coordinates;
  }

  private int calculateVertexParity(double[] coordinates) {
    int count = 0;
    for (double val : coordinates) {
      if (val == 1.0) {
        count++;
      }
    }
    return (count % 2 == 1) ? 1 : 0;
  }
}
