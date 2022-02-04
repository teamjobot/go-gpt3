package gpt3

import (
	"crypto/rand"
	"math/big"
	"strings"
)

// Float32Ptr converts a float to an *float32 as a convenience
func Float32Ptr(f float32) *float32 {
	return &f
}

// float32PtrDefault returns int ptr if not nil otherwise creates int with default value and returns pointer.
func float32PtrDefault(i *float32, defaultValue float32) *float32 {
	if i == nil {
		i = new(float32)
		*i = defaultValue
	}

	return i
}

// IntPtr converts an integer to an *int as a convenience
func IntPtr(i int) *int {
	return &i
}

// intPtrDefault returns int ptr if not nil otherwise creates int with default value and returns pointer.
func intPtrDefault(i *int, defaultValue int) *int {
	if i == nil {
		i = new(int)
		*i = defaultValue
	}

	return i
}

func trimStr(input *string) string {
	if input == nil {
		return ""
	}

	return strings.TrimSpace(*input)
}

func intPtrRand(min, max int) *int {
	r := int(random(int64(min), int64(max)))
	return &r
}

func float32Rand(min, max float32) float32 {
	r := random(int64(min*100), int64(max*100))
	r2 := float32(r) / float32(100)
	return r2
}

func float32PtrRand(min, max float32) *float32 {
	r2 := float32Rand(min, max)
	return &r2
}

func random(min, max int64) int64 {
	bg := big.NewInt(max - min)

	n, err := rand.Int(rand.Reader, bg)
	if err != nil {
		return min
	}

	return n.Int64() + min
}
