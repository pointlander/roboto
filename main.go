// Copyright 2026 The Roboto Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Size is the size of the tape
	Size = 16
	// InputWidth is the width of the input
	InputWidth = 256
	// EmbeddingWidth is the width of the embedding
	EmbeddingWidth = 5
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Action is an action
type Action byte

const (
	// ActionNone is the do nothing action
	ActionNone Action = iota
	// ActionUp is the up action
	ActionUp
	// ActionDown is the down action
	ActionDown
	// ActionLeft is the left action
	ActionLeft
	// ActionRight is the right action
	ActionRight
	// ActionCount number of actinos
	ActionCount
)

// String the string version of action
func (a Action) String() string {
	switch a {
	case ActionNone:
		return "None"
	case ActionUp:
		return "Up"
	case ActionDown:
		return "Down"
	case ActionLeft:
		return "Left"
	case ActionRight:
		return "Right"
	}
	return "Unknown"
}

// State is a state
type State struct {
	Image     []float64
	Embedding []float64
	Action    Action
}

// States is a set of states
type States struct {
	Buffer []State
	Head   int
	Rng    *rand.Rand
	X      int
	Y      int
	Set    tf64.Set
}

// NewStates creates a new states
func NewStates(rng *rand.Rand, size, width int, length int) States {
	buffer := make([]State, length)
	for i := range buffer {
		buffer[i].Image = make([]float64, Size*Size)
		for j := range buffer[i].Image {
			buffer[i].Image[j] = float64(rng.Intn(2))
		}
		buffer[i].Action = Action(rng.Intn(int(ActionCount)))
	}
	set := tf64.NewSet()
	set.Add("i", width, length)
	set.Add("w0", size, size)
	set.Add("b0", size)
	set.Add("w1", 2*size, size)
	set.Add("b1", size)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	return States{
		Buffer: buffer,
		Rng:    rng,
		Set:    set,
	}
}

// LearnEmbedding learns the embedding
func (s *States) LearnEmbedding(size, width int) {
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	others.Add("x", size, len(s.Buffer))
	x := others.ByName["x"]
	head := s.Head
	for {
		x.X = append(x.X, s.Buffer[head].Image...)
		head = (head + 1) % len(s.Buffer)
		if head == s.Head {
			break
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}
	set := s.Set
	l0 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), others.Get("x")), set.Get("b0")))
	l1 := tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(l1)))
	loss := tf64.Avg(tf64.Quadratic(l1, sa))

	for iteration := range 8 {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(iteration, l)
	}

	I := set.ByName["i"]
	head = s.Head
	for i := range s.Buffer {
		s.Buffer[head].Embedding = I.X[i*width : (i+1)*width]
		head = (head + 1) % len(s.Buffer)
		if head == s.Head {
			break
		}
	}
}

// Markov is a markov state
type Markov [2]Action

// Iterate iterates the markov state
func (m *Markov) Iterate(s Action) {
	m[0], m[1] = m[1], s
}

// Bucket is the entry in a markov model
type Bucket struct {
	Entries []State
}

// Model is a markov model
type Model struct {
	Model [2]map[Markov]*Bucket
	Root  *Bucket
}

// NewModel creates a new model
func NewModel() (model Model) {
	for i := range model.Model {
		model.Model[i] = make(map[Markov]*Bucket)
	}
	model.Root = &Bucket{}
	return model
}

// Set sets an entry
func (m *Model) Set(markov Markov, entry State) {
	for i := range 2 {
		bucket := m.Model[i][markov]
		if bucket == nil {
			bucket = &Bucket{}
		}
		bucket.Entries = append(bucket.Entries, entry)
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
	m.Root.Entries = append(m.Root.Entries, entry)
}

// Get gets an entry
func (m *Model) Get(markov Markov) *Bucket {
	for i := range 2 {
		bucket := m.Model[i][markov]
		if bucket != nil {
			return bucket
		}
		markov[i] = 0
	}
	return m.Root
}

// Next finds the next symbol
func (s *States) Next() Action {
	const (
		Eta = 1.0e-3
	)
	s.LearnEmbedding(InputWidth, EmbeddingWidth)

	dot := func(a, b []float64) float64 {
		x := 0.0
		for i, value := range a {
			x += value * b[i]
		}
		return x
	}

	cs := func(a, b []float64) float64 {
		ab := dot(a, b)
		aa := dot(a, a)
		bb := dot(b, b)
		if aa <= 0 {
			return 0
		}
		if bb <= 0 {
			return 0
		}
		return ab / (math.Sqrt(aa) * math.Sqrt(bb))
	}

	rng := rand.New(rand.NewSource(1))

	var markov Markov
	model := NewModel()
	head := s.Head
	for {
		model.Set(markov, s.Buffer[head])
		markov.Iterate(s.Buffer[head].Action)
		head = (head + 1) % len(s.Buffer)
		if head == s.Head {
			break
		}
	}
	type Result struct {
		Actions []Action
		Cost    float64
	}
	process := func(markov Markov) Result {
		symbols := make([]Action, 0, 33)
		current := s.Buffer[s.Head].Embedding
		cost := 0.0
		for range 33 {
			bucket := model.Get(markov)
			d := make([]float64, len(bucket.Entries))
			sum := 0.0
			for i, entry := range bucket.Entries {
				x := cs(current, entry.Embedding) + 1
				d[i] = x
				sum += x
			}
			total, selected, index := 0.0, rng.Float64(), 0
			for i, value := range d {
				total += value / sum
				if selected < total {
					index = i
					break
				}
			}
			symbol := bucket.Entries[index].Action
			symbols = append(symbols, symbol)
			cost += d[index] / sum
			current = bucket.Entries[index].Embedding
			markov.Iterate(symbol)
		}
		return Result{
			Actions: symbols,
			Cost:    cost,
		}
	}
	results := make([]Result, 0, 256*1024)
	for range 256 * 1024 {
		result := process(markov)
		results = append(results, result)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Cost > results[j].Cost
	})
	fmt.Println(results[0].Actions[0])

	return results[0].Actions[0]
}

func (s *States) Update() error {
	next := s.Next()
	sample := s.Rng.Intn(2 * int(ActionCount))
	if sample < int(ActionCount) {
		next = Action(sample)
	}
	n := (s.Head + 1) % len(s.Buffer)
	s.Buffer[n].Image = s.Buffer[s.Head].Image
	switch next {
	case ActionNone:
	case ActionUp:
		s.Y = (s.Y - 1 + Size) % Size
	case ActionDown:
		s.Y = (s.Y + 1) % Size
	case ActionLeft:
		s.X = (s.X - 1 + Size) % Size
	case ActionRight:
		s.X = (s.X + 1) % Size
	}
	if s.Buffer[n].Image[s.Y*Size+s.X] >= .5 {
		s.Buffer[n].Image[s.Y*Size+s.X] = 0.0
	} else {
		s.Buffer[n].Image[s.Y*Size+s.X] = 1.0
	}
	s.Buffer[n].Action = next
	s.Head = n

	x, y := ebiten.CursorPosition()
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		if s.Buffer[n].Image[y*Size+x] >= .5 {
			s.Buffer[n].Image[y*Size+x] = 0.0
		} else {
			s.Buffer[n].Image[y*Size+x] = 1.0
		}
	}
	return nil
}

func (s *States) Draw(screen *ebiten.Image) {
	input := s.Buffer[s.Head].Image
	for y := range Size {
		for x := range Size {
			if input[y*Size+x] >= .5 {
				screen.Set(x, y, color.RGBA{0, 0, 0xFF, 0xFF})
			} else {
				screen.Set(x, y, color.RGBA{0, 0, 0, 0})
			}
		}
	}
}

func (s *States) Layout(outsideWidth, outsideHeight int) (int, int) {
	return Size, Size
}

func main() {
	rng := rand.New(rand.NewSource(1))
	states := NewStates(rng, InputWidth, EmbeddingWidth, 33)
	ebiten.SetWindowSize(256, 256)
	ebiten.SetWindowTitle("Neuron")
	err := ebiten.RunGame(&states)
	if err != nil {
		panic(err)
	}
}
