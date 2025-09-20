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

func Softmax(x *[]float64) []float64 {
	output := make([]float64, len(*x))
	sum := 0.0
	for i, v := range *x {
		output[i] = math.Exp(v)
		sum += output[i]
	}
	for i := range output {
		output[i] /= sum
	}
	return output
}
