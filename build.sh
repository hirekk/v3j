#!/bin/bash

echo "🚀 V3J Build Script"
echo "==================="

case "$1" in
    "compile")
        echo "📦 Compiling Java files..."
        javac -d build/classes src/main/java/*.java
        ;;
    "run")
        echo "🏃‍♂️ Running Main class..."
        java -cp build/classes Main
        ;;
    "test")
        echo "🧪 Running tests with Gradle..."
        ./gradlew test
        ;;
    "build")
        echo "🔨 Building with Gradle..."
        ./gradlew build
        ;;
    "format")
        echo "✨ Formatting code with Google Java Style..."
        ./gradlew spotlessApply
        ;;
    "check-format")
        echo "🔍 Checking code formatting..."
        ./gradlew spotlessCheck
        ;;
    "clean")
        echo "🧹 Cleaning compiled files..."
        rm -rf build/classes
        ./gradlew clean
        ;;
    *)
        echo "Usage: $0 {compile|run|test|build|format|check-format|clean}"
        echo ""
        echo "Commands:"
        echo "  compile      - Compile Java files (just javac)"
        echo "  run          - Run the program (just java)"
        echo "  test         - Run tests (requires Gradle)"
        echo "  build        - Full build with Gradle"
        echo "  format       - Format code with Google Java Style"
        echo "  check-format - Check if code follows formatting rules"
        echo "  clean        - Clean everything"
        echo ""
        echo "Examples:"
        echo "  $0 compile  # Quick compile with javac"
        echo "  $0 run      # Quick run with java"
        echo "  $0 format   # Format code with Google style"
        echo "  $0 test     # Run tests with Gradle"
        ;;
esac
