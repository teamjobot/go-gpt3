package gpt3

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
