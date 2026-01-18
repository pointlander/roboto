// Copyright 2026 The Roboto Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Fisher is the fisher iris data
type Fisher struct {
	Measures  []float64
	Embedding []float64
	Label     string
	L         byte
	Index     int
}

// LearnEmbedding learns the embedding
func LearnEmbedding(iris []Fisher, size, width int) []Fisher {
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	length := len(iris)
	cp := make([]Fisher, length)
	copy(cp, iris)
	others.Add("x", size, len(cp))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}

	set := tf64.NewSet()
	set.Add("i", width, len(cp))
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

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	l0 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), others.Get("x")), set.Get("b0")))
	l1 := tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(l1)))
	loss := tf64.Avg(tf64.Quadratic(l1, sa))

	for iteration := range 256 {
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
			return nil
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
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
	}

	return cp
}

// Markov is a markov state
type Markov [2]byte

// Iterate iterates the markov state
func (m *Markov) Iterate(s byte) {
	m[0], m[1] = m[1], s
}

// Bucket is the entry in a markov model
type Bucket struct {
	Entries []Fisher
}

// Model is a markov model
type Model struct {
	Model [2]map[Markov]*Bucket
}

// NewModel creates a new model
func NewModel() (model Model) {
	for i := range model.Model {
		model.Model[i] = make(map[Markov]*Bucket)
	}
	return model
}

// Set sets an entry
func (m *Model) Set(markov Markov, entry Fisher) {
	for i := range 2 {
		bucket := m.Model[i][markov]
		if bucket == nil {
			bucket = &Bucket{}
		}
		bucket.Entries = append(bucket.Entries, entry)
		m.Model[i][markov] = bucket
		markov[i] = 0
	}
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
	return nil
}

// Next finds the next symbol
func Next(input []byte) byte {
	const (
		Eta = 1.0e-3
	)
	book := make([]Fisher, 0, 8)
	for i, symbol := range input {
		b := Fisher{
			Measures: make([]float64, 256),
			L:        symbol,
			Index:    i,
		}
		b.Measures[symbol] = 1
		input = append(input, symbol)
		book = append(book, b)
	}
	fmt.Println(string(input))
	width := 5
	cp := LearnEmbedding(book, 256, width)

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
	for _, entry := range cp {
		model.Set(markov, entry)
		markov.Iterate(entry.L)
	}
	type Result struct {
		Symbols []byte
		Cost    float64
	}
	process := func(markov Markov) Result {
		symbols := make([]byte, 0, 33)
		current := cp[len(cp)-1].Embedding
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
			symbol := bucket.Entries[index].L
			symbols = append(symbols, symbol)
			cost += d[index] / sum
			current = bucket.Entries[index].Embedding
			markov.Iterate(symbol)
		}
		return Result{
			Symbols: symbols,
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
	fmt.Println("`" + string(input) + "`")
	fmt.Println("`" + string(results[0].Symbols) + "`")

	return results[0].Symbols[0]
}

func main() {

}
