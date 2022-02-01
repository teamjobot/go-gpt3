package gpt3

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

type InterviewArgs struct {
	JobTitle       *string
	JobDescription *string
}

type InterviewInput struct {
	Engine    string
	MaxTokens *int
	Prompt    string
	Temp      *float32
}

type InterviewResponse struct {
	Input InterviewInput
	Duration time.Duration
	Questions []InterviewQuestion
}

type InterviewQuestion struct {
	Index    int
	Question string
}

func (r *InterviewResponse) QuestionText() string {
	var sb strings.Builder

	for index, r := range r.Questions {
		sb.WriteString(r.Question)

		if index+1 < len(r.Question) {
			sb.WriteString("\n\n")
		}
	}

	return sb.String()
}

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

func (c *client) InterviewQuestions(ctx context.Context, args InterviewArgs) (*InterviewResponse, error) {
	start := time.Now()
	jobTitle := trimStr(args.JobTitle)
	jobDesc := trimStr(args.JobDescription)

	if len(jobTitle) == 0 && len(jobDesc) == 0 {
		return nil, errors.New("must specify a job title or description")
	}

	engine := DavinciInstructEngine
	prompt := getInterviewPrompt(jobTitle, jobDesc)

	request := CompletionRequest{
		MaxTokens:   IntPtr(64),
		Prompt:      []string{prompt},
		Temperature: Float32Ptr(0.8),
	}

	resp, err := c.CompletionWithEngine(ctx, engine, request)

	if err != nil {
		return nil, err
	}

	result := &InterviewResponse{
		Input: InterviewInput{
			Engine:    engine,
			MaxTokens: request.MaxTokens,
			Prompt:    prompt,
			Temp:      request.Temperature,
		},
	}

	// Will only be one result max really
	for _, ch := range resp.Choices {
		items := parseInterviewChoice(ch)

		if items != nil {
			result.Questions = append(result.Questions, items...)
		}
	}

	result.Duration = time.Since(start)

	return result, err
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

			ques = strings.TrimSpace(ques)

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
