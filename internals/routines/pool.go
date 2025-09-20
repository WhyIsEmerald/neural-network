package routines

import (
	"runtime"
	"sync"
)

var GlobalPool *Pool

func init() {
	GlobalPool = NewPool(runtime.NumCPU())
}

// Pool manages a collection of workers.
type Pool struct {
	Tasks   chan func()
	workers []*Worker

	workerWg sync.WaitGroup
	taskWg   sync.WaitGroup
}

func NewPool(numWorkers int) *Pool {
	p := &Pool{
		Tasks: make(chan func(), numWorkers*2),
	}

	for i := 1; i <= numWorkers; i++ {
		worker := NewWorker(i, p.Tasks)
		p.workers = append(p.workers, worker)
		worker.Start(&p.workerWg)
	}

	return p
}

func (p *Pool) AddTask(task func()) {
	p.taskWg.Add(1)
	p.Tasks <- func() {
		defer p.taskWg.Done()
		task()
	}
}

func (p *Pool) WaitAll() {
	p.taskWg.Wait()
}

func (p *Pool) StopAll() {
	for _, worker := range p.workers {
		worker.Stop()
	}
	p.workerWg.Wait()
}
