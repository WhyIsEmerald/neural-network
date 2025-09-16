package math

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

func (m *Matrix) Get(row, col int) float64 {
	return m.Data[row*m.Cols+col]
}

func (m *Matrix) Set(row, col int, value float64) {
	m.Data[row*m.Cols+col] = value
}

func (m *Matrix) Add(m2 *Matrix) (*Matrix, error) {
	if m.Rows != m2.Rows || m.Cols != m2.Cols {
		return nil, errors.New("matrix dimensions do not match")
	}
	result := NewMatrix(m.Rows, m.Cols, make([]float64, m.Rows*m.Cols))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Set(i, j, m.Get(i, j)+m2.Get(i, j))
		}
	}
	return result, nil
}

func (m1 *Matrix) Subtract(m2 *Matrix) (*Matrix, error) {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return nil, errors.New("Matrix dimensions do not match")
	}
	out := NewMatrix(m1.Rows, m1.Cols, make([]float64, m1.Rows*m1.Cols))
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			out.Set(i, j, m1.Get(i, j)-m2.Get(i, j))
		}
	}
	return out, nil
}

func (m1 *Matrix) DotProduct(m2 *Matrix) (*Matrix, error) {
	r1 := m1.Rows
	r2 := m2.Rows
	c1 := m1.Cols
	c2 := m2.Cols
	if c1 != r2 {
		return nil, errors.New("Matrix dimensions are incompatable")
	}
	out := NewMatrix(r1, c2, make([]float64, r1*c2))

	pool := routines.GlobalPool

	for i := 0; i < r1; i++ {
		row := i
		pool.AddTask(func() {
			for j := 0; j < c2; j++ {
				sum := 0.0
				for k := 0; k < c1; k++ {
					sum += m1.Get(row, k) * m2.Get(k, j)
				}
				out.Set(row, j, sum)
			}
		})
	}

	pool.WaitAll()
	return out, nil
}

func (m *Matrix) Transpose() *Matrix {
	out := NewMatrix(m.Cols, m.Rows, make([]float64, m.Rows*m.Cols))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Set(j, i, m.Get(i, j))
		}
	}
	return out
}
