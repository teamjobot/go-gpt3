package gpt3

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const (
	InterviewEngine = "text-davinci-001"
)

type InterviewArgs struct {
	Cap            *int
	JobTitle       *string
	JobDescription *string
}

type InterviewInput struct {
	Engine           string
	FrequencyPenalty float32
	MaxTokens        *int
	N                *float32
	PresencePenalty  float32
	Prompt           string
	Temp             *float32
	TopP             *float32
}

type InterviewResponse struct {
	Input     InterviewInput
	Duration  time.Duration
	Questions []InterviewQuestion
}

type InterviewQuestion struct {
	Index    int
	Question string
}

func (r *InterviewResponse) HasQuestions() bool {
	return r != nil && r.Questions != nil && len(r.Questions) > 0
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

	engine := InterviewEngine
	prompt := getInterviewPrompt(jobTitle, jobDesc)

	/*
		Frequence penalty:
		Lowers the chances of a word being selected again the more times that word has already been used.

		Presence Penalty:
		Presence penalty does not consider how frequently a word has been used, but just if the word exists in the text.
		This helps to make it less repetitive and seem more natural.

		TopP and Temp use one or the other, set other to 1...
		Controls diversity via nucleus sampling; 0.5 means half of al likelihood-weighted options are considered.
		"Top P provides better control for applications in which GPT-3 is expected to generate text with accuracy and
		correctness, while Temperature works best for those applications in which original, creative or even amusing
		responses are sought."

		MaxTokens (might make client arg later):
		- 512 can generate about 51 questions but takes over 20 seconds.
		- 128 about 12 ques in 5+sec.
		- 75 seems good for about 5ish questions ~3 sec
	*/
	request := CompletionRequest{
		FrequencyPenalty: .75,

		MaxTokens: IntPtr(175),

		// Just 1, there are multiple answers in one block stream
		N: Float32Ptr(1),

		PresencePenalty: .7,

		Prompt: []string{prompt},

		Temperature: Float32Ptr(1),
		TopP:        Float32Ptr(0.85),
	}

	cap := 5
	if args.Cap != nil {
		cap = *args.Cap
	}

	resp, err := c.CompletionWithEngine(ctx, engine, request)

	if err != nil {
		return nil, err
	}

	// Trying not to expose GPT-3 types to insulate caller but we are repeating some things w/that
	result := &InterviewResponse{
		Input: InterviewInput{
			Engine:           engine,
			FrequencyPenalty: request.FrequencyPenalty,
			MaxTokens:        request.MaxTokens,
			N:                request.N,
			PresencePenalty:  request.PresencePenalty,
			Prompt:           prompt,
			Temp:             request.Temperature,
			TopP:             request.TopP,
		},
	}

	// Will only be one result max really
	for _, ch := range resp.Choices {
		items := parseInterviewChoice(ch)

		if items != nil {
			// result.Questions = append(result.Questions, items...)
			for _, qu := range items {
				if len(result.Questions) == cap {
					break
				}
				result.Questions = append(result.Questions, qu)
			}
		}
	}

	result.Duration = time.Since(start)

	return result, err
}

func parseText(question string) string {
	ques := strings.TrimSpace(question)

	if strings.HasPrefix(ques, "-") {
		ques = ques[1:]
	}

	// Sometimes question results are numbered 1), 2), etc. which we want to strip
	pos := strings.Index(ques, ") ")

	// i.e. "1)" through "99)"
	if pos > -1 && pos <= 2 {
		tmp := ques[0:pos]
		_, err := strconv.Atoi(tmp)

		if err == nil {
			//ques = ques[pos:len(tmp)-pos]
			ques = ques[pos+2:]
		}
	}

	return strings.TrimSpace(ques)
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
			ques := parseText(part)

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
