package math

import (
	"testing"
)

func TestNewMatrix(t *testing.T) {
	m := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	if m.Rows != 2 {
		t.Errorf("m.Rows = %d; want 2", m.Rows)
	}
	if m.Cols != 3 {
		t.Errorf("m.Cols = %d; want 3", m.Cols)
	}
}

func TestMatrix_Get(t *testing.T) {
	m := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	if m.Get(1, 1) != 5 {
		t.Errorf("m.Get(1, 1) = %f; want 5", m.Get(1, 1))
	}
}

func TestMatrix_Set(t *testing.T) {
	m := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	m.Set(1, 1, 10)
	if m.Get(1, 1) != 10 {
		t.Errorf("m.Get(1, 1) = %f; want 10", m.Get(1, 1))
	}
}

func TestMatrix_Add(t *testing.T) {
	m1 := NewMatrix(2, 2, []float64{1, 2, 3, 4})
	m2 := NewMatrix(2, 2, []float64{5, 6, 7, 8})
	result, err := m1.Add(m2)
	if err != nil {
		t.Errorf("m1.Add(m2) returned an error: %v", err)
	}
	expected := NewMatrix(2, 2, []float64{6, 8, 10, 12})
	for i := 0; i < 4; i++ {
		if result.Data[i] != expected.Data[i] {
			t.Errorf("result.Data[%d] = %f; want %f", i, result.Data[i], expected.Data[i])
		}
	}
}

func TestMatrix_Subtract(t *testing.T) {
	m1 := NewMatrix(2, 2, []float64{5, 6, 7, 8})
	m2 := NewMatrix(2, 2, []float64{1, 2, 3, 4})
	result, err := m1.Subtract(m2)
	if err != nil {
		t.Errorf("m1.Subtract(m2) returned an error: %v", err)
	}
	expected := NewMatrix(2, 2, []float64{4, 4, 4, 4})
	for i := 0; i < 4; i++ {
		if result.Data[i] != expected.Data[i] {
			t.Errorf("result.Data[%d] = %f; want %f", i, result.Data[i], expected.Data[i])
		}
	}
}

func TestMatrix_DotProduct(t *testing.T) {
	m1 := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	m2 := NewMatrix(3, 2, []float64{7, 8, 9, 10, 11, 12})
	result, err := m1.DotProduct(m2)
	if err != nil {
		t.Errorf("m1.DotProduct(m2) returned an error: %v", err)
	}
	expected := NewMatrix(2, 2, []float64{58, 64, 139, 154})
	for i := 0; i < 4; i++ {
		if result.Data[i] != expected.Data[i] {
			t.Errorf("result.Data[%d] = %f; want %f", i, result.Data[i], expected.Data[i])
		}
	}
}

func TestMatrix_Transpose(t *testing.T) {
	m := NewMatrix(2, 3, []float64{1, 2, 3, 4, 5, 6})
	result := m.Transpose()
	expected := NewMatrix(3, 2, []float64{1, 4, 2, 5, 3, 6})
	for i := 0; i < 6; i++ {
		if result.Data[i] != expected.Data[i] {
			t.Errorf("result.Data[%d] = %f; want %f", i, result.Data[i], expected.Data[i])
		}
	}
}
