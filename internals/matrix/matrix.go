package matrix

import (
	"errors"

	"github.com/whyisemerald/neural_network/internals/routines"
)

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func NewMatrix(rows, cols int, data []float64) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

func Get(row, col int, m *Matrix) float64 {
	return m.Data[row*m.Cols+col]
}

func Set(row, col int, value float64, m *Matrix) {
	m.Data[row*m.Cols+col] = value
}

func Add(m1, m2, out *Matrix) error {
	return elementWiseOp(m1, m2, out, func(a, b float64) float64 {
		return a + b
	})
}

func Subtract(m1, m2, out *Matrix) error {
	return elementWiseOp(m1, m2, out, func(a, b float64) float64 {
		return a - b
	})
}

func DotProduct(m1, m2, out *Matrix) error {
	r1 := m1.Rows
	r2 := m2.Rows
	c1 := m1.Cols
	c2 := m2.Cols
	if c1 != r2 {
		return errors.New("Matrix dimensions are incompatable")
	}
	if len(m1.Data) > 1000 {
		pool := routines.GlobalPool

		for i := 0; i < r1; i++ {
			row := i
			pool.AddTask(func() {
				for j := 0; j < c2; j++ {
					sum := 0.0
					for k := 0; k < c1; k++ {
						sum += Get(row, k, m1) * Get(k, j, m2)
					}
					Set(row, j, sum, out)
				}
			})
		}

		pool.WaitAll()
	} else {
		for i := 0; i < r1; i++ {
			for j := 0; j < c2; j++ {
				sum := 0.0
				for k := 0; k < c1; k++ {
					sum += Get(i, k, m1) * Get(k, j, m2)
				}
				Set(i, j, sum, out)
			}
		}
	}
	return nil
}
func Transpose(m *Matrix) *Matrix {
	out := NewMatrix(m.Cols, m.Rows, make([]float64, m.Rows*m.Cols))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			Set(j, i, Get(i, j, m), out)
		}
	}
	return out
}

func MultiplyElementWise(m1, m2, out *Matrix) error {
	return elementWiseOp(m1, m2, out, func(a, b float64) float64 {
		return a * b
	})
}
func MultiplyScalar(m *Matrix, scalar float64, out *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			Set(i, j, Get(i, j, m)*scalar, out)
		}
	}
}

func elementWiseOp(m1, m2, out *Matrix, op func(float64, float64) float64) error {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return errors.New("Matrix dimensions do not match")
	}
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			Set(i, j, op(Get(i, j, m1), Get(i, j, m2)), out)
		}
	}
	return nil
}
func ApplyFunction(m *Matrix, fn func(float64) float64, out *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			Set(i, j, fn(Get(i, j, m)), out)
		}
	}
}
