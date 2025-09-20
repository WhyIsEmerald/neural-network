package math

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Softmax computes the normalized exponential function for a vector.
// To improve numerical stability, the maximum value is subtracted from all elements before exponentiation.
func Softmax(x *[]float64) []float64 {
	output := make([]float64, len(*x))
	maxVal := math.Inf(-1)
	for _, v := range *x {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := 0.0
	for i, v := range *x {
		output[i] = math.Exp(v - maxVal)
		sum += output[i]
	}
	for i := range output {
		output[i] /= sum
	}
	return output
}
