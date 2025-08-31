# V3J - VIII in Java

## Octonion-based neural network architecture

An attempt to train octonion-valued neural network without backpropagation (and perhaps arbitrary activation function?)...

- ... but it's actually quaternions--for now.
- aaand it doesn't really work... (for now?)

[![codecov](https://codecov.io/github/hirekk/v3j/branch/main/graph/badge.svg?token=QD3MLN3EF1)](https://codecov.io/github/hirekk/v3j)

## The idea

Leverage the rich geometry of S^7 (or S^3 for the quaternion PoC) to do away with backpropagation.

Eventually, take advantage of non-associativity of octonion multiplication to get rid of arbitrary activation functions.

## System Requirements

- **Java 24+** - JDK required
- **Gradle** - Included via wrapper (`./gradlew`)

## Quick Start

### 1. Install Java
```bash
# macOS with Homebrew
brew install openjdk@24

# Or download from:
# https://adoptium.net/ (Eclipse Temurin)
# https://www.oracle.com/java/technologies/downloads/ (Oracle)
```

### 2. Build the Project
```bash
./gradlew build
```

### 3. Create Executable JAR
```bash
./gradlew jar
```

### 4. Test the CLI
```bash
# The v3j script is already included in the repository
# Test that it works (requires the JAR to be built first)
./v3j --help
```

### 5. Train the perceptron on 2D XOR data.
```bash
# Generate a 2D XOR dataset
./v3j generate exact -d 2 -o data/xor/exact2d.csv

# Train on the dataset
./v3j train --data data/xor/exact2d.csv
```

## Usage

### Basic Commands
```bash
# General help
./v3j --help

# Generate commands
./v3j generate --help
./v3j generate exact --help
./v3j generate fuzzy --help

# Training
./v3j train --help
```

### Examples

**Generate 2D exact XOR dataset:**
```bash
./v3j generate exact -d 2 -o data/xor/exact2d.csv
```

**Generate 3D fuzzy XOR dataset:**
```bash
./v3j generate fuzzy -d 3 -c 10 -v 0.15 -o data/xor/fuzzy3d.csv
```

**Train the model:**
```bash
./v3j train --data data/xor/exact2d.csv
```

### Alternative: Using Gradle
```bash
./gradlew run --args="generate exact -d 2 -o data/xor/exact2d.csv"
./gradlew run --args="train --data data/xor/exact2d.csv"
```

## Advanced Setup

### Using v3j from Any Directory

**Option 1: Temporary PATH (Current Session Only)**
```bash
# Add to current shell session (doesn't persist)
export PATH="$PWD:$PATH"  # bash/zsh
# OR
set -gx PATH $PWD $PATH   # fish

# Now you can use v3j from anywhere in this session
v3j --help
```

**Option 2: Install to Standard Location**
```bash
# Copy to /usr/local/bin (requires sudo)
sudo cp build/libs/v3j.jar /usr/local/bin/v3j
sudo chmod +x /usr/local/bin/v3j

# Create wrapper script
sudo tee /usr/local/bin/v3j > /dev/null << 'EOF'
#!/bin/bash
java -jar /usr/local/bin/v3j.jar "$@"
EOF

# Now v3j works system-wide
v3j --help
```

**Option 3: Use from Project Directory**
```bash
# Always run from project directory
cd /path/to/v3j
./v3j --help
```

## Build Commands

```bash
./gradlew build      # Build everything
./gradlew test       # Run tests
./gradlew clean      # Clean build artifacts
./gradlew jar        # Create executable JAR
```

## Project Structure

```
v3j/
├── build.gradle             # Gradle config with Java 24, JUnit, Picocli, Spotless
├── gradlew                  # Gradle wrapper
├── v3j                      # Shell script wrapper (included)
├── src/
│   ├── data/               # Data generation package
│   │   └── README.md       # Detailed data package documentation
│   └── math/               # Mathematical abstractions package
│       └── README.md       # Detailed math package documentation
└── build/                   # Build output (auto-generated)
    └── libs/
        └── v3j.jar         # Executable JAR (after running ./gradlew jar)
```

## Project Features

- **Modern Java 24** with records, pattern matching
- **Strategy Pattern** for different generation approaches
- **Builder Pattern** for fluent API design
- **Custom CSV I/O** without external dependency based on known data format
- **CLI interface** using Picocli with short aliases
- **Code formatting** with Spotless
- **JUnit 5** testing framework

## Documentation

For detailed usage instructions and examples, see the package-specific READMEs:
- [Data Package Documentation](src/data/README.md)
- [Math Package Documentation](src/math/README.md)
