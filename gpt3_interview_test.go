package gpt3

import (
	"testing"
)

func TestStripLeadingNumbers(t *testing.T) {
	type testCase struct {
		name     string
		call     func() string
		expected string
	}

	testCases := []testCase{
		{
			"Paren Space",
			func() string {
				return stripLeadingNumbers("3) How has NetSuite helped you manage your construction business?")
			},
			"How has NetSuite helped you manage your construction business?",
		},
		{
			"Period Space",
			func() string {
				return stripLeadingNumbers("3. How has NetSuite helped you manage your construction business?")
			},
			"How has NetSuite helped you manage your construction business?",
		},
		{
			"No Space",
			func() string {
				return stripLeadingNumbers("3.How has NetSuite helped you manage your construction business?")
			},
			"How has NetSuite helped you manage your construction business?",
		},
		{
			"Period embedded parens",
			func() string {
				return stripLeadingNumbers("3. What NAS Solutions (enterprise and scale-out) are you familiar with?")
			},
			"What NAS Solutions (enterprise and scale-out) are you familiar with?",
		},
		{
			"No hits",
			func() string {
				return stripLeadingNumbers("How has NetSuite helped you manage your construction business?")
			},
			"How has NetSuite helped you manage your construction business?",
		},
		{
			"Spaces trimmed",
			func() string {
				return stripLeadingNumbers(" How has NetSuite helped you manage your construction business? ")
			},
			"How has NetSuite helped you manage your construction business?",
		},
		{
			"Empty string",
			func() string {
				return stripLeadingNumbers("")
			},
			"",
		},
	}

	//3. What NAS Solutions (enterprise and scale-out) are you familiar with?

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.call()

			if result != tc.expected {
				t.Errorf("\nGot: '%s'\nExpected: '%s'", result, tc.expected)
				return
			}
		})
	}
}
