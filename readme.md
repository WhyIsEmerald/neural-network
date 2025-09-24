# Go Neural Network

A simple neural network implemented in Go to solve the XOR problem. This project demonstrates the implementation of a feedforward neural network with backpropagation.

## Features

- Custom matrix operations for efficient computation.
- Sigmoid activation function for non-linearity.
- Multi-threaded support for large matrix operations.
- Save and load functionality for trained models.
- Example usage with the XOR dataset.

## Prerequisites

- Go 1.24.4 or later.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/whyisemerald/neural_network.git
   cd neural_network
   ```

2. Run the neural network:
   ```bash
   go run main.go
   ```

## Example Output

After training the network on the XOR dataset, the program will output predictions for the following inputs:

- `[0, 1]`
- `[1, 0]`
- `[1, 1]`
- `[0, 0]`

Example output:
```
Training took: 5.730567922s
Output for [0,1]: 0.9956646461578814
Output for [1,0]: 0.9956639091536774
Output for [1,1]: 0.005173442587170633
Output for [0,0]: 0.00505068755246391
Forward took: 8.998µs
```

## Project Structure

- `main.go`: Entry point for the neural network.
- `internals/math`: Contains mathematical functions like sigmoid and softmax.
- `internals/matrix`: Matrix operations for neural network computations.
- `internals/network`: Neural network layers and training logic.
- `internals/routines`: Multi-threading utilities for parallel computation.

## License

This project is open-source and available under the MIT License.
