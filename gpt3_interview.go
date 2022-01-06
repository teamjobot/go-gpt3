package gpt3

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

func formatInterviewInput(input string) string {
	output := newLineRe.ReplaceAllString(input, " ")
	output = strings.ReplaceAll(output, "•", "")
	return output
}

func getInterviewPrompt(jobTitle, jobDesc string) string {
	var prompt string

	if len(jobTitle) > 0 && len(jobDesc) > 0 {
		prompt = fmt.Sprintf(
			"Create a list of questions for my interview with a %s, %s",
			formatInterviewInput(jobTitle),
			formatInterviewInput(jobDesc))
	} else if len(jobTitle) > 0 {
		prompt = fmt.Sprintf("Create a list of questions for my interview with a %s", formatInterviewInput(jobTitle))
	} else if len(jobDesc) > 0 {
		prompt = fmt.Sprintf(
			"Create a list of questions for my interview with a job description of %s",
			formatInterviewInput(jobDesc))
	}

	return prompt
}

func (c *client) InterviewQuestions(ctx context.Context, args InterviewArgs) ([]InterviewQuestion, error) {
	jobTitle := trimStr(args.JobTitle)
	jobDesc := trimStr(args.JobDescription)

	if len(jobTitle) == 0 && len(jobDesc) == 0 {
		return nil, errors.New("must specify a job title or description")
	}

	prompt := getInterviewPrompt(jobTitle, jobDesc)

	resp, err := c.CompletionWithEngine(
		ctx,
		DavinciInstructEngine,
		CompletionRequest{
			MaxTokens:   IntPtr(64),
			Prompt:      []string{prompt},
			Temperature: Float32Ptr(0.8),
		})

	if err != nil {
		return nil, err
	}

	var data []InterviewQuestion

	// Will only be one result max really
	for _, ch := range resp.Choices {
		items := parseInterviewChoice(ch)

		if items != nil {
			data = append(data, items...)
		}
	}

	return data, err
}

func parseInterviewChoice(ch CompletionResponseChoice) []InterviewQuestion {
	var data []InterviewQuestion

	if len(ch.Text) == 0 {
		return nil
	}

	parts := strings.Split(ch.Text, "\n")

	for _, part := range parts {
		// Last question can be truncated. Might also need to check ch.FinishReason for length later
		if len(part) > 0 && strings.HasSuffix(part, "?") {
			ques := part

			// TODO: occasionally the responses are numbered in the text
			if strings.HasPrefix(ques, "-") {
				ques = ques[1:]
			}

			data = append(data, InterviewQuestion{
				Index:    len(data) + 1,
				Question: ques,
			})
		}
	}

	if len(data) == 0 {
		return nil
	}

	return data
}
