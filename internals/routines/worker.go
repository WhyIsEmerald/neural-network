package routines

import "sync"

type Worker struct {
	ID   int
	Task chan func()
	Quit chan bool
}

func NewWorker(id int, task chan func()) *Worker {
	return &Worker{
		ID:   id,
		Task: task,
		Quit: make(chan bool),
	}
}

func (w *Worker) Start(wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case task := <-w.Task:
				task()
			case <-w.Quit:
				return
			}
		}
	}()
}

func (w *Worker) Stop() {
	go func() {
		w.Quit <- true
	}()
}

func (w *Worker) IsIdle() bool {
	select {
	case <-w.Task:
		return false
	default:
		return true
	}
}
