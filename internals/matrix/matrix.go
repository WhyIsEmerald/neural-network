package matrix

import (
	"errors"
	"runtime"

	"github.com/whyisemerald/neural_network/internals/routines"
)

const PARALLEL_THRESHOLD = 422500

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
	if len(m1.Data) > PARALLEL_THRESHOLD {
		pool := routines.GlobalPool
		numWorkers := runtime.NumCPU()
		chunkSize := (r1 + numWorkers - 1) / numWorkers

		for i := 0; i < r1; i += chunkSize {
			startRow := i
			endRow := i + chunkSize
			if endRow > r1 {
				endRow = r1
			}
			pool.AddTask(func() {
				for row := startRow; row < endRow; row++ {
					for j := 0; j < c2; j++ {
						sum := 0.0
						for k := 0; k < c1; k++ {
							sum += Get(row, k, m1) * Get(k, j, m2)
						}
						Set(row, j, sum, out)
					}
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
	if len(m.Data) > PARALLEL_THRESHOLD {
		pool := routines.GlobalPool
		numWorkers := runtime.NumCPU()
		chunkSize := (m.Rows + numWorkers - 1) / numWorkers

		for i := 0; i < m.Rows; i += chunkSize {
			startRow := i
			endRow := i + chunkSize
			if endRow > m.Rows {
				endRow = m.Rows
			}
			pool.AddTask(func() {
				for row := startRow; row < endRow; row++ {
					for j := 0; j < m.Cols; j++ {
						Set(j, row, Get(row, j, m), out)
					}
				}
			})
		}
		pool.WaitAll()
	} else {
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				Set(j, i, Get(i, j, m), out)
			}
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
	if len(m.Data) > PARALLEL_THRESHOLD {
		pool := routines.GlobalPool
		numWorkers := runtime.NumCPU()
		chunkSize := (m.Rows + numWorkers - 1) / numWorkers

		for i := 0; i < m.Rows; i += chunkSize {
			startRow := i
			endRow := i + chunkSize
			if endRow > m.Rows {
				endRow = m.Rows
			}
			pool.AddTask(func() {
				for row := startRow; row < endRow; row++ {
					for j := 0; j < m.Cols; j++ {
						Set(row, j, Get(row, j, m)*scalar, out)
					}
				}
			})
		}
		pool.WaitAll()
	} else {
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				Set(i, j, Get(i, j, m)*scalar, out)
			}
		}
	}
}

func elementWiseOp(m1, m2, out *Matrix, op func(float64, float64) float64) error {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return errors.New("Matrix dimensions do not match")
	}
	if len(m1.Data) > PARALLEL_THRESHOLD {
		pool := routines.GlobalPool
		numWorkers := runtime.NumCPU()
		chunkSize := (m1.Rows + numWorkers - 1) / numWorkers

		for i := 0; i < m1.Rows; i += chunkSize {
			startRow := i
			endRow := i + chunkSize
			if endRow > m1.Rows {
				endRow = m1.Rows
			}
			pool.AddTask(func() {
				for row := startRow; row < endRow; row++ {
					for j := 0; j < m1.Cols; j++ {
						Set(row, j, op(Get(row, j, m1), Get(row, j, m2)), out)
					}
				}
			})
		}
		pool.WaitAll()
	} else {
		for i := 0; i < m1.Rows; i++ {
			for j := 0; j < m1.Cols; j++ {
				Set(i, j, op(Get(i, j, m1), Get(i, j, m2)), out)
			}
		}
	}
	return nil
}
func ApplyFunction(m *Matrix, fn func(float64) float64, out *Matrix) {
	if len(m.Data) > PARALLEL_THRESHOLD {
		pool := routines.GlobalPool
		numWorkers := runtime.NumCPU()
		chunkSize := (m.Rows + numWorkers - 1) / numWorkers

		for i := 0; i < m.Rows; i += chunkSize {
			startRow := i
			endRow := i + chunkSize
			if endRow > m.Rows {
				endRow = m.Rows
			}
			pool.AddTask(func() {
				for row := startRow; row < endRow; row++ {
					for j := 0; j < m.Cols; j++ {
						Set(row, j, fn(Get(row, j, m)), out)
					}
				}
			})
		}
		pool.WaitAll()
	} else {
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				Set(i, j, fn(Get(i, j, m)), out)
			}
		}
	}
}