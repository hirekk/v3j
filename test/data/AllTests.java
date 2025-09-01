/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

/**
 * Test suite runner for all data package tests. This class serves as a container for organizing
 * related tests.
 */
@DisplayName("Data Package Tests")
class AllTests {

  @Nested
  @DisplayName("DataPoint Tests")
  class DataPointTests extends DataPointTest {}

  @Nested
  @DisplayName("DatasetGenerator Tests")
  class DatasetGeneratorTests extends DatasetGeneratorTest {}

  @Nested
  @DisplayName("Dataset Tests")
  class DatasetTests extends DatasetTest {}
}
