// Copyright 2026 The Roboto Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"testing"
)

func BenchmarkSystem(b *testing.B) {
	print = false
	rng := rand.New(rand.NewSource(1))
	states := NewStates(rng, InputWidth, EmbeddingWidth, 33)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		states.Update()
	}
}
