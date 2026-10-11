package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"crypto/rand"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"net/http"
	"os/signal"
	"strings"
	"time"
)

func main() {
	verbose := flag.Bool("verbose", false, "enable verbose logging")
	flag.Parse()

	if *verbose {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags | log.Lshortfile)
		log.Println("Verbose logging enabled")
	} else {
		log.SetOutput(os.Stdout)
		log.SetFlags(0)
		log.SetPrefix("")
	}

	client := anthropic.NewClient()
	if *verbose {
		log.Println("Anthropic client initialized")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	// Read input separately so an idle prompt can select on cancellation.
	scanner := bufio.NewScanner(os.Stdin)
	messages := make(chan string)
	go func() {
		defer close(messages)
		for scanner.Scan() {
			select {
			case messages <- scanner.Text():
			case <-ctx.Done():
				return
			}
		}
	}()
	getUserMessage := func() (string, bool) {
		select {
		case <-ctx.Done():
			return "", false
		case text, ok := <-messages:
			return text, ok
		}
	}

	session, tools, err := connectParallel(ctx, parallelEndpoint, &http.Client{Transport: projectTransport{http.DefaultTransport}})
	if err != nil {
		fmt.Fprintln(os.Stderr, "Connect to Parallel:", err)
		os.Exit(1)
	}
	defer session.Close()
	if *verbose {
		log.Printf("Initialized %d tools", len(tools))
	}
	agent := NewAgent(&client, getUserMessage, tools, *verbose)
	err = agent.Run(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func NewAgent(
	client *anthropic.Client,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
	verbose bool,
) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
		verbose:        verbose,
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
	verbose        bool
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	if a.verbose {
		log.Println("Starting chat session with tools enabled")
	}
	fmt.Println("Chat with Claude (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			if a.verbose {
				log.Println("User input ended, breaking from chat loop")
			}
			break
		}

		// Skip empty messages
		if userInput == "" {
			if a.verbose {
				log.Println("Skipping empty message")
			}
			continue
		}

		if a.verbose {
			log.Printf("User input received: %q", userInput)
		}

		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		if a.verbose {
			log.Printf("Sending message to Claude, conversation length: %d", len(conversation))
		}

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			if a.verbose {
				log.Printf("Error during inference: %v", err)
			}
			return err
		}
		conversation = append(conversation, message.ToParam())

		// Keep processing until Claude stops using tools
		for {
			// Collect all tool uses and their results
			var toolResults []anthropic.ContentBlockParamUnion
			var hasToolUse bool

			if a.verbose {
				log.Printf("Processing %d content blocks from Claude", len(message.Content))
			}

			for _, content := range message.Content {
				switch content.Type {
				case "text":
					fmt.Printf("\u001b[93mClaude\u001b[0m: %s\n", content.Text)
				case "tool_use":
					hasToolUse = true
					toolUse := content.AsToolUse()
					if a.verbose {
						log.Printf("Tool use detected: %s with input: %s", toolUse.Name, string(toolUse.Input))
					}
					fmt.Printf("\u001b[96mtool\u001b[0m: %s(%s)\n", toolUse.Name, string(toolUse.Input))

					// Find and execute the tool
					var toolResult string
					var toolError error
					var toolFound bool
					for _, tool := range a.tools {
						if tool.Name == toolUse.Name {
							if a.verbose {
								log.Printf("Executing tool: %s", tool.Name)
							}
							toolResult, toolError = tool.Function(ctx, toolUse.Input)
							fmt.Printf("\u001b[92mresult\u001b[0m: %s\n", toolResult)
							if toolError != nil {
								fmt.Printf("\u001b[91merror\u001b[0m: %s\n", toolError.Error())
							}
							if a.verbose {
								if toolError != nil {
									log.Printf("Tool execution failed: %v", toolError)
								} else {
									log.Printf("Tool execution successful, result length: %d chars", len(toolResult))
								}
							}
							toolFound = true
							break
						}
					}

					if !toolFound {
						toolError = fmt.Errorf("tool '%s' not found", toolUse.Name)
						fmt.Printf("\u001b[91merror\u001b[0m: %s\n", toolError.Error())
					}

					// Add tool result to collection
					if toolError != nil {
						toolResults = append(toolResults, anthropic.NewToolResultBlock(toolUse.ID, toolError.Error(), true))
					} else {
						toolResults = append(toolResults, anthropic.NewToolResultBlock(toolUse.ID, toolResult, false))
					}
				}
			}

			// If there were no tool uses, we're done
			if !hasToolUse {
				break
			}

			// Send all tool results back and get Claude's response
			if a.verbose {
				log.Printf("Sending %d tool results back to Claude", len(toolResults))
			}
			toolResultMessage := anthropic.NewUserMessage(toolResults...)
			conversation = append(conversation, toolResultMessage)

			// Get Claude's response after tool execution
			message, err = a.runInference(ctx, conversation)
			if err != nil {
				if a.verbose {
					log.Printf("Error during followup inference: %v", err)
				}
				return err
			}
			conversation = append(conversation, message.ToParam())

			if a.verbose {
				log.Printf("Received followup response with %d content blocks", len(message.Content))
			}

			// Continue loop to process the new message
		}
	}

	if a.verbose {
		log.Println("Chat session ended")
	}
	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	anthropicTools := []anthropic.ToolUnionParam{}
	for _, tool := range a.tools {
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: tool.InputSchema,
			},
		})
	}

	if a.verbose {
		log.Printf("Making API call to Claude with model: %s and %d tools", anthropic.ModelClaudeOpus4_6, len(anthropicTools))
	}

	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeOpus4_6,
		MaxTokens: int64(1024),
		Messages:  conversation,
		Tools:     anthropicTools,
	})

	if a.verbose {
		if err != nil {
			log.Printf("API call failed: %v", err)
		} else {
			log.Printf("API call successful, response received")
		}
	}

	return message, err
}

type ToolDefinition struct {
	Name        string                         `json:"name"`
	Description string                         `json:"description"`
	InputSchema anthropic.ToolInputSchemaParam `json:"input_schema"`
	Function    func(context.Context, json.RawMessage) (string, error)
}

// This optional workshop stage connects the existing tool registry to MCP.
const parallelEndpoint = "https://search.parallel.ai/mcp"
const projectUserAgent = "how-to-build-a-coding-agent/1.0 (+https://github.com/ghuntley/how-to-build-a-coding-agent)"
const parallelTimeout = 60 * time.Second

type projectTransport struct{ base http.RoundTripper }

func (t projectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set("User-Agent", projectUserAgent)
	return t.base.RoundTrip(clone)
}

func connectParallel(ctx context.Context, endpoint string, client *http.Client) (*mcp.ClientSession, []ToolDefinition, error) {
	startup, cancel := context.WithTimeout(ctx, parallelTimeout)
	defer cancel()
	session, err := mcp.NewClient(&mcp.Implementation{Name: "how-to-build-a-coding-agent", Version: "1.0"}, nil).Connect(startup, &mcp.StreamableClientTransport{Endpoint: endpoint, HTTPClient: client}, nil)
	if err != nil {
		return nil, nil, err
	}
	tools, err := parallelTools(startup, session)
	if err != nil {
		session.Close()
		return nil, nil, err
	}
	return session, tools, nil
}

func parallelTools(ctx context.Context, session *mcp.ClientSession) ([]ToolDefinition, error) {
	// Use one trusted identifier throughout this conversation for free-tier accounting.
	id := make([]byte, 16)
	if _, err := rand.Read(id); err != nil {
		return nil, err
	}
	sessionID := fmt.Sprintf("%x", id)
	var tools []ToolDefinition
	for tool, err := range session.Tools(ctx, nil) {
		if err != nil {
			return nil, err
		}
		if tool.Name != "web_search" && tool.Name != "web_fetch" {
			continue
		}
		raw, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return nil, err
		}
		var schema anthropic.ToolInputSchemaParam
		if err := json.Unmarshal(raw, &schema); err != nil {
			return nil, err
		}
		name := tool.Name
		tools = append(tools, ToolDefinition{Name: name, Description: tool.Description, InputSchema: schema,
			Function: func(ctx context.Context, input json.RawMessage) (string, error) {
				var args map[string]any
				if err := json.Unmarshal(input, &args); err != nil {
					return "", err
				}
				if args == nil {
					return "", fmt.Errorf("tool arguments must be an object")
				}
				args["session_id"] = sessionID
				args["model_name"] = string(anthropic.ModelClaudeOpus4_6)
				callCtx, cancel := context.WithTimeout(ctx, parallelTimeout)
				defer cancel()
				result, err := session.CallTool(callCtx, &mcp.CallToolParams{Name: name, Arguments: args})
				if err != nil {
					return "", err
				}
				var text []string
				for _, content := range result.Content {
					if block, ok := content.(*mcp.TextContent); ok {
						text = append(text, block.Text)
					}
				}
				output := strings.Join(text, "\n")
				if result.IsError {
					return "", fmt.Errorf("%s: %s", name, output)
				}
				if output == "" {
					return "", fmt.Errorf("%s returned no text", name)
				}
				return output, nil
			}})
	}
	if len(tools) != 2 {
		return nil, fmt.Errorf("Parallel must expose web_search and web_fetch")
	}
	return tools, nil
}
