package math_test

import (
	"math"
	"testing"

	nnmath "github.com/WhyIsEmerald/neural_network/internals/math"
)

func TestSigmoid(t *testing.T) {
	if nnmath.Sigmoid(0) != 0.5 {
		t.Errorf("Sigmoid(0) = %f; want 0.5", nnmath.Sigmoid(0))
	}
}

func TestRelu(t *testing.T) {
	if nnmath.Relu(10) != 10 {
		t.Errorf("Relu(10) = %f; want 10", nnmath.Relu(10))
	}
	if nnmath.Relu(-10) != 0 {
		t.Errorf("Relu(-10) = %f; want 0", nnmath.Relu(-10))
	}
}

func TestSoftmax(t *testing.T) {
	input := []float64{1, 2, 3}
	expected := []float64{0.09003057, 0.24472847, 0.66524096}
	output := nnmath.Softmax(input)
	for i := range output {
		if math.Abs(output[i]-expected[i]) > 1e-8 {
			t.Errorf("Softmax(%v)[%d] = %f; want %f", input, i, output[i], expected[i])
		}
	}
}
