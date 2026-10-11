package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type recordedRequest struct{ method, url, userAgent, authorization string }
type recorderTransport struct {
	base     http.RoundTripper
	mu       sync.Mutex
	requests []recordedRequest
}

func (r *recorderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.requests = append(r.requests, recordedRequest{req.Method, req.URL.String(), req.UserAgent(), req.Header.Get("Authorization")})
	r.mu.Unlock()
	return r.base.RoundTrip(req)
}
func (r *recorderTransport) check(t *testing.T, endpoint string) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	posts := 0
	for _, req := range r.requests {
		if req.url != endpoint || req.userAgent != projectUserAgent || req.authorization != "" {
			t.Errorf("unexpected MCP request: %+v", req)
		}
		if req.method == "POST" {
			posts++
		}
	}
	if posts < 5 {
		t.Errorf("want discovery and both tool calls; got %d POSTs", posts)
	}
}
func selectTool(t *testing.T, tools []ToolDefinition, name string) ToolDefinition {
	t.Helper()
	for _, tool := range tools {
		if tool.Name == name {
			return tool
		}
	}
	t.Fatalf("missing %s", name)
	return ToolDefinition{}
}

func TestParallelAgentLoop(t *testing.T) {
	var mu sync.Mutex
	var calls []string
	var sessionIDs []string
	server := mcp.NewServer(&mcp.Implementation{Name: "fixture", Version: "1"}, nil)
	for _, name := range []string{"web_search", "web_fetch"} {
		name := name
		schema := map[string]any{"type": "object", "properties": map[string]any{"objective": map[string]any{"type": "string"}}, "required": []string{"objective"}}
		server.AddTool(&mcp.Tool{Name: name, Description: "Fixture " + name, InputSchema: schema}, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			var args map[string]any
			if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
				return nil, err
			}
			if args["model_name"] != string(anthropic.ModelClaudeOpus4_6) {
				t.Error("untrusted model name")
			}
			id, _ := args["session_id"].(string)
			if len(id) != 32 {
				t.Errorf("invalid session ID: %q", id)
			}
			mu.Lock()
			calls = append(calls, name)
			sessionIDs = append(sessionIDs, id)
			mu.Unlock()
			if args["objective"] == "fail" {
				return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "fixture tool failure"}}}, nil
			}
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: name + " source https://go.dev/blog/context"}}}, nil
		})
	}
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, &mcp.StreamableHTTPOptions{Stateless: true})
	endpoint := httptest.NewServer(handler)
	defer endpoint.Close()
	recorder := &recorderTransport{base: http.DefaultTransport}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	session, tools, err := connectParallel(ctx, endpoint.URL, &http.Client{Transport: projectTransport{recorder}})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	schema, _ := json.Marshal(selectTool(t, tools, "web_search").InputSchema)
	if !strings.Contains(string(schema), `"required":["objective"]`) {
		t.Fatalf("lost required schema: %s", schema)
	}
	// The real Anthropic SDK speaks to a deterministic fixture; the real Agent.Run
	// must advertise tools, dispatch both, and return results before the final reply.
	inferenceCalls := 0
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools []struct {
				Name string `json:"name"`
			}
			Messages []struct {
				Content []struct {
					Type    string
					Content json.RawMessage
					IsError bool `json:"is_error"`
				} `json:"content"`
			}
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/json")
		inferenceCalls++
		if len(body.Tools) != 2 {
			t.Errorf("want 2 tools, got %d", len(body.Tools))
		}
		if inferenceCalls == 1 {
			fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-6","stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"tool_use","id":"search_1","name":"web_search","input":{"objective":"context","session_id":"model-value","model_name":"model-value"}},{"type":"tool_use","id":"fetch_1","name":"web_fetch","input":{"objective":"context"}}]}`)
		} else {
			results := body.Messages[len(body.Messages)-1].Content
			if len(results) != 2 {
				t.Errorf("want 2 results, got %d", len(results))
			}
			for _, result := range results {
				if result.Type != "tool_result" || result.IsError || !strings.Contains(string(result.Content), "https://go.dev/blog/context") {
					t.Errorf("bad result: %+v", result)
				}
			}
			fmt.Fprint(w, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-4-6","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"Context cancels work; source https://go.dev/blog/context"}]}`)
		}
	}))
	defer api.Close()
	client := anthropic.NewClient(option.WithBaseURL(api.URL), option.WithAPIKey("fixture"), option.WithMaxRetries(0))
	inputs := 0
	agent := NewAgent(&client, func() (string, bool) { inputs++; return "Search and fetch context documentation", inputs == 1 }, tools, false)
	if err := agent.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if inferenceCalls != 2 {
		t.Errorf("want follow-up inference, got %d", inferenceCalls)
	}
	mu.Lock()
	if strings.Join(calls, ",") != "web_search,web_fetch" || sessionIDs[0] != sessionIDs[1] {
		t.Errorf("bad dispatch/session: %v %v", calls, sessionIDs)
	}
	mu.Unlock()
	_, err = selectTool(t, tools, "web_search").Function(ctx, json.RawMessage(`{"objective":"fail"}`))
	if err == nil || !strings.Contains(err.Error(), "fixture tool failure") {
		t.Errorf("lost MCP tool error: %v", err)
	}
	for _, input := range []string{`null`, `[]`, `{"broken"`} {
		if _, err := tools[0].Function(ctx, json.RawMessage(input)); err == nil {
			t.Errorf("accepted invalid arguments %s", input)
		}
	}
	canceled, stop := context.WithCancel(ctx)
	stop()
	if _, err := tools[0].Function(canceled, json.RawMessage(`{"objective":"context"}`)); err == nil {
		t.Error("ignored cancellation")
	}
	recorder.check(t, endpoint.URL)
}

func TestParallelLive(t *testing.T) {
	if os.Getenv("PARALLEL_LIVE") != "1" {
		t.Skip("set PARALLEL_LIVE=1 for anonymous live MCP validation")
	}
	// No key or saved credential loader is used by this MCP client.
	t.Setenv("PARALLEL_API_KEY", "")
	recorder := &recorderTransport{base: http.DefaultTransport}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	session, tools, err := connectParallel(ctx, parallelEndpoint, &http.Client{Transport: projectTransport{recorder}})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	for _, call := range []struct{ name, input string }{
		{"web_search", `{"objective":"Find official Go context cancellation documentation","search_queries":["Go context cancellation documentation"]}`},
		{"web_fetch", `{"urls":["https://go.dev/blog/context"],"objective":"Explain Go context cancellation"}`},
	} {
		output, err := selectTool(t, tools, call.name).Function(ctx, json.RawMessage(call.input))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(output, "https://go.dev/") || !strings.Contains(strings.ToLower(output), "context") {
			t.Fatalf("no useful source output: %s", output)
		}
		t.Logf("%s returned %d bytes with Go context source", call.name, len(output))
	}
	recorder.check(t, parallelEndpoint)
}
