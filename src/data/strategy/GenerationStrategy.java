package data.strategy;

import data.DataPoint;
import java.util.List;

public interface GenerationStrategy {
  List<DataPoint> generateDataset(int numDimensions);
}
