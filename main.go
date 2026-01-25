// Copyright 2026 The Roboto Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/pointlander/gradient/tf32"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Size is the size of the tape
	Size = 8
	// InputWidth is the width of the input
	InputWidth = 2 * Size * Size
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

//go:embed books/*
var Books embed.FS

// Book is a book
type Book struct {
	Name string
	Text []byte
}

// LoadBooks loads books
func LoadBooks() []Book {
	books := []Book{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}
	load := func(book string) []byte {
		file, err := Books.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		return data
	}
	for i := range books {
		books[i].Text = load(fmt.Sprintf("books/%s", books[i].Name))
	}
	return books
}

var print = true

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
	// ActionFlip flips the pixel
	ActionFlip
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

// Int is an int type
type Int interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// Shard is a markov model entry
type Shard[T Int] struct {
	Embedding []float32
	Action    T
}

// State is a state
type State[T Int] struct {
	Shard[T]
	Image []float32
}

// States is a set of states
type States[T Int] struct {
	Size   int
	Buffer []State[T]
	Head   int
	Rng    *rand.Rand
	X      int
	Y      int
	Set    tf32.Set
	Model  Model[T]
	Markov Markov[T]
	lock   sync.Mutex
}

// NewStates creates a new states
func NewStates[T Int](rng *rand.Rand, size, width int, length int) States[T] {
	buffer := make([]State[T], length)
	for i := range buffer {
		buffer[i].Image = make([]float32, size)
	}
	set := tf32.NewSet()
	set.Add("i", width, length)
	set.Add("w0", size, size)
	set.Add("b0", size)
	set.Add("w1", 2*size, size)
	set.Add("b1", size)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float32, len(w.X))
		}
	}

	return States[T]{
		Size:   size,
		Buffer: buffer,
		Rng:    rng,
		Set:    set,
		Model:  NewModel[T](),
	}
}

// LearnEmbedding learns the embedding
func (s *States[T]) LearnEmbedding() {
	rng := rand.New(rand.NewSource(1))
	others := tf32.NewSet()
	others.Add("x", s.Size, len(s.Buffer))
	x := others.ByName["x"]
	head := s.Head
	for {
		head = (head + 1) % len(s.Buffer)
		x.X = append(x.X, s.Buffer[head].Image...)
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
	l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("x")), set.Get("b0")))
	l1 := tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1"))
	sa := tf32.T(tf32.Mul(tf32.Dropout(tf32.Square(set.Get("i")), dropout), tf32.T(l1)))
	loss := tf32.Avg(tf32.Quadratic(l1, sa))

	for iteration := range Iterations {
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		set.Zero()
		others.Zero()
		l := tf32.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += float64(d * d)
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
				g := d * float32(scaling)
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		if print {
			fmt.Println(iteration, l)
		}
	}

	ii := set.ByName["i"]
	head = s.Head
	for i := range ii.S[1] {
		cp := make([]float32, ii.S[0])
		copy(cp, ii.X[i*ii.S[0]:(i+1)*ii.S[0]])
		s.Buffer[head].Embedding = cp
		head = (head + 1) % len(s.Buffer)
		if head == s.Head {
			break
		}
	}
}

// Markov is a markov state
type Markov[T Int] [2]T

// Iterate iterates the markov state
func (m *Markov[T]) Iterate(s T) {
	m[0], m[1] = m[1], s
}

// Bucket is the entry in a markov model
type Bucket[T Int] struct {
	Entries []Shard[T]
	Head    int
}

// Model is a markov model
type Model[T Int] struct {
	Model [4]map[Markov[T]]*Bucket[T]
	Root  *Bucket[T]
}

// NewModel creates a new model
func NewModel[T Int]() (model Model[T]) {
	for i := range model.Model {
		model.Model[i] = make(map[Markov[T]]*Bucket[T])
	}
	model.Root = &Bucket[T]{}
	model.Root.Entries = make([]Shard[T], 128)
	return model
}

// Set sets an entry
func (m *Model[T]) Set(markov Markov[T], entry State[T]) {
	for i := range 2 {
		bucket := m.Model[i][markov]
		if bucket == nil {
			bucket = &Bucket[T]{}
			bucket.Entries = make([]Shard[T], 256)
		}
		bucket.Head = (bucket.Head + 1) % len(bucket.Entries)
		bucket.Entries[bucket.Head] = entry.Shard
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
	m.Root.Head = (m.Root.Head + 1) % len(m.Root.Entries)
	m.Root.Entries[m.Root.Head] = entry.Shard
}

// Get gets an entry
func (m *Model[T]) Get(markov Markov[T]) *Bucket[T] {
	for i := range 2 {
		bucket := m.Model[i][markov]
		if bucket != nil {
			return bucket
		}
		markov[i] = 0
	}
	return m.Root
}

func cs(a, b []float32) float32 {
	ab := tf32.Dot(a, b)
	aa := tf32.Dot(a, a)
	bb := tf32.Dot(b, b)
	if aa <= 0 {
		return 0
	}
	if bb <= 0 {
		return 0
	}
	return ab / (float32(math.Sqrt(float64(aa))) * float32(math.Sqrt(float64(bb))))
}

// Next finds the next symbol
func (s *States[T]) Next() T {
	const (
		Eta = 1.0e-3
	)
	s.LearnEmbedding()

	rng := rand.New(rand.NewSource(1))

	markov := s.Markov
	head := s.Head
	s.Markov.Iterate(T(s.Buffer[(head+1)%len(s.Buffer)].Action))
	for {
		head = (head + 1) % len(s.Buffer)
		s.Model.Set(markov, s.Buffer[head])
		markov.Iterate(s.Buffer[head].Action)
		if head == s.Head {
			break
		}
	}
	type Result struct {
		Actions []T
		Cost    float32
	}
	process := func(markov Markov[T]) Result {
		symbols := make([]T, 0, 33)
		current := s.Buffer[s.Head].Embedding
		cost := float32(0.0)
		for range 33 {
			bucket := s.Model.Get(markov)
			sum := float32(0.0)
			count := 0
			for _, entry := range bucket.Entries {
				if entry.Embedding == nil {
					continue
				}
				count++
			}
			d := make([]float32, count)
			count = 0
			for _, entry := range bucket.Entries {
				if entry.Embedding == nil {
					continue
				}
				x := cs(current, entry.Embedding) + 1
				d[count] = x
				sum += x
				count++
			}
			total, selected, index := float32(0.0), rng.Float32(), 0
			count = 0
			for i, entry := range bucket.Entries {
				if entry.Embedding == nil {
					continue
				}
				total += d[count] / sum
				if selected < total {
					index = i
					break
				}
				count++
			}
			symbol := bucket.Entries[index].Action
			symbols = append(symbols, symbol)
			cost += d[count] / sum
			current = bucket.Entries[index].Embedding
			markov.Iterate(symbol)
		}
		return Result{
			Actions: symbols,
			Cost:    cost,
		}
	}
	results := make([]Result, 0, 1024)
	for range 1024 {
		result := process(markov)
		results = append(results, result)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Cost > results[j].Cost
	})
	if print {
		fmt.Println(results[0].Actions[0])
	}

	return results[0].Actions[0]
}

func (s *States[T]) Update() error {
	s.lock.Lock()
	defer s.lock.Unlock()
	next := s.Next()
	sample := s.Rng.Intn(8 * int(ActionCount))
	if sample < int(ActionCount) {
		next = T(sample)
	}
	n := (s.Head + 1) % len(s.Buffer)
	copy(s.Buffer[n].Image, s.Buffer[s.Head].Image)
	s.Buffer[n].Image[Size*Size+s.Y*Size+s.X] = 0.0
	switch Action(next) {
	case ActionNone:
	case ActionUp:
		s.Y = (s.Y - 1 + Size) % Size
	case ActionDown:
		s.Y = (s.Y + 1) % Size
	case ActionLeft:
		s.X = (s.X - 1 + Size) % Size
	case ActionRight:
		s.X = (s.X + 1) % Size
	case ActionFlip:
		if s.Buffer[n].Image[s.Y*Size+s.X] >= .5 {
			s.Buffer[n].Image[s.Y*Size+s.X] = 0.0
		} else {
			s.Buffer[n].Image[s.Y*Size+s.X] = 1.0
		}
	}
	s.Buffer[n].Image[Size*Size+s.Y*Size+s.X] = 1.0
	s.Buffer[n].Action = next
	s.Head = n

	x, y := ebiten.CursorPosition()
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		s.Buffer[n].Image[y*Size+x] = 0.0
	}
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight) {
		s.Buffer[n].Image[y*Size+x] = 1.0
	}
	return nil
}

func (s *States[T]) Draw(screen *ebiten.Image) {
	s.lock.Lock()
	defer s.lock.Unlock()
	input := s.Buffer[s.Head].Image
	for y := range Size {
		for x := range Size {
			if input[Size*Size+y*Size+x] > .5 {
				if input[y*Size+x] >= .5 {
					screen.Set(x, y, color.RGBA{0xFF, 0, 0xFF, 0xFF})
				} else {
					screen.Set(x, y, color.RGBA{0xFF, 0, 0, 0})
				}
			} else {
				if input[y*Size+x] >= .5 {
					screen.Set(x, y, color.RGBA{0, 0, 0xFF, 0xFF})
				} else {
					screen.Set(x, y, color.RGBA{0, 0, 0, 0})
				}
			}
		}
	}
}

func (s *States[T]) Layout(outsideWidth, outsideHeight int) (int, int) {
	return Size, Size
}

var (
	// FlagText text mode
	FlagText = flag.Bool("text", false, "text mode")
)

func main() {
	flag.Parse()

	if *FlagText {
		rng := rand.New(rand.NewSource(1))
		s := NewStates[byte](rng, 256, EmbeddingWidth, 64)
		books := LoadBooks()
		book := books[1]
		print = false
		for i, symbol := range book.Text[:4*1024] {
			s.Next()
			n := (s.Head + 1) % len(s.Buffer)
			for i := range s.Buffer[n].Image {
				s.Buffer[n].Image[i] = 0
			}
			s.Buffer[n].Image[symbol] = 1.0
			s.Buffer[n].Action = symbol
			s.Head = n
			fmt.Println(i)
		}
		for {
			symbol := s.Next()
			n := (s.Head + 1) % len(s.Buffer)
			for i := range s.Buffer[n].Image {
				s.Buffer[n].Image[i] = 0
			}
			s.Buffer[n].Image[symbol] = 1.0
			s.Buffer[n].Action = symbol
			s.Head = n
			fmt.Printf("%c", symbol)
		}
		return
	}

	rng := rand.New(rand.NewSource(1))
	states := NewStates[Action](rng, InputWidth, EmbeddingWidth, 33)
	for i := range states.Buffer {
		for j := range states.Buffer[i].Image[:Size*Size] {
			states.Buffer[i].Image[j] = float32(rng.Intn(2))
		}
		states.Buffer[i].Action = Action(rng.Intn(int(ActionCount)))
	}
	ebiten.SetWindowSize(256, 256)
	ebiten.SetWindowTitle("Neuron")
	err := ebiten.RunGame(&states)
	if err != nil {
		panic(err)
	}
}
