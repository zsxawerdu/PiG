package coding

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/MichaelKinsy/PiG/coding/extension"
	"github.com/MichaelKinsy/PiG/coding/extension/host/subprocess"
	icodingagent "github.com/MichaelKinsy/PiG/internal/codingagent"
)

func setupNodeRefreshHost(t *testing.T, services *AgentSessionServices, extPath string) (*subprocess.Host, *extension.Extension) {
	t.Helper()
	session, err := NewSession(services, SessionOptions{
		Model:            &ai.Model{ID: "primary", Provider: &scriptedProvider{}},
		NoSession:        true,
		SkipBuiltinTools: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close() })

	host := subprocess.NewHost(t.TempDir())
	t.Cleanup(func() { host.Shutdown("test done") })
	host.SetProviderCallbacks(services.Registry().RegisterExtensionProvider, services.Registry().UnregisterProvider)
	bridge := subprocess.NewUIBridge(func() {})
	detach := icodingagent.WireModelOperations(bridge, icodingagent.ModelOperationBindings{
		CurrentModel:  session.Model,
		ModelLookup:   services.ModelRuntime().GetModel,
		ModelCatalog:  func(...string) []*ai.Model { return services.ModelRuntime().GetModels() },
		Registry:      services.Registry().ModelRegistry,
		ModelBuilder:  func(spec string) (*ai.Model, error) { return BuildModel(spec, services) },
		SessionHandle: session,
	})
	t.Cleanup(detach)
	host.SetUIBridge(bridge)

	ext, err := host.Load(t.Context(), subprocess.ExtConfig{Name: "dynamic-provider", Source: extPath, Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	applyQueuedRegistrations(host)
	return host, ext
}

func TestNodeProviderConfigRefreshModelsReplacesModels(t *testing.T) {
	extDir := t.TempDir()
	extPath := filepath.Join(extDir, "dynamic-provider.mjs")
	err := os.WriteFile(extPath, []byte(`
import { readFileSync } from "node:fs";
import { join } from "node:path";

export default function (pi) {
  let calls = 0;
  let lastCtx = null;
  let failNext = false;
  let waitSignal = false;
  let agentDir = "";

  pi.registerProvider("dynamic", {
    baseUrl: "https://example.invalid/v1",
    api: "openai-completions",
    apiKey: "configured",
    streamSimple: (model, _ctx, _opts) => {
      const ZERO = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
      const usage = { ...ZERO, totalTokens: 0, cost: { ...ZERO, total: 0 } };
      const message = { role: "assistant", api: model.api, provider: model.provider, model: model.id, content: [{ type: "text", text: "streamed:" + model.id }], usage, stopReason: "stop", timestamp: 1 };
      return { async *[Symbol.asyncIterator]() { yield { type: "start", partial: message }; yield { type: "done", reason: "stop", message }; }, result: async () => message };
    },
    models: [
      { id: "old", name: "Old Model", input: ["text"], contextWindow: 1000, maxTokens: 100 },
    ],
    refreshModels: async (ctx) => {
      calls++;
      lastCtx = {
        allowNetwork: ctx.allowNetwork,
        force: ctx.force,
        baseUrl: ctx.baseUrl,
        hasCredential: Boolean(ctx.credential),
        stored: ctx.stored ?? null,
      };
      if (waitSignal) {
        await new Promise((resolve, reject) => {
          if (ctx.signal.aborted) {
            reject(new Error("aborted"));
            return;
          }
          ctx.signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
        });
      }
      if (failNext) {
        throw new Error("refresh exploded");
      }
      if (ctx.allowNetwork) {
        let updated = false;
        const accepted = await ctx.publish({
          persist: { models: [{ id: "persisted-m" }] },
          update: () => {
            if (agentDir) {
              const store = readFileSync(join(agentDir, "models-store.json"), "utf8");
              if (!store.includes("persisted-m")) {
                throw new Error("publication was not persisted before update");
              }
            }
            updated = true;
          },
        });
        if (!accepted || !updated) {
          throw new Error("publish did not accept or run update");
        }
        return [
          { id: "new", name: "New Model", input: ["text"], contextWindow: 2000, maxTokens: 200 },
        ];
      }
      if (ctx.stored?.models?.some((m) => m.id === "persisted-m")) {
        return [
          { id: "restored", name: "Restored Model", input: ["text"], contextWindow: 3000, maxTokens: 300 },
        ];
      }
      return undefined;
    },
  });

  pi.registerCommand("set-agent-dir", {
    description: "Set agentDir",
    handler: async (args) => { agentDir = args.trim(); },
  });
  pi.registerCommand("set-flags", {
    description: "Set test flags",
    handler: async (args) => {
      const parsed = JSON.parse(args);
      failNext = Boolean(parsed.failNext);
      waitSignal = Boolean(parsed.waitSignal);
    },
  });
  pi.registerCommand("check-calls", {
    description: "Check refreshModels call count",
    handler: async (_args, ctx) => {
      if (calls === 0) throw new Error("refreshModels was never called");
      if (!ctx.modelRegistry.find("dynamic", "new")) {
        throw new Error("refreshed model not visible in ctx.modelRegistry");
      }
    },
  });
  pi.registerCommand("check-ctx", {
    description: "Check last context",
    handler: async (args) => {
      const want = JSON.parse(args);
      for (const [k, v] of Object.entries(want)) {
        if (JSON.stringify(lastCtx?.[k]) !== JSON.stringify(v)) {
          throw new Error("lastCtx." + k + " = " + JSON.stringify(lastCtx?.[k]) + ", want " + JSON.stringify(v));
        }
      }
    },
  });
  pi.registerCommand("late-register", {
    description: "Register a late dynamic provider",
    handler: async () => {
      pi.registerProvider("late-dynamic", {
        baseUrl: "https://late.invalid/v1",
        api: "openai-completions",
        apiKey: "configured",
        models: [{ id: "late-old", name: "Late Old", input: ["text"], contextWindow: 1000, maxTokens: 100 }],
        refreshModels: async () => [
          { id: "late-new", name: "Late New", input: ["text"], contextWindow: 4000, maxTokens: 400 },
        ],
      });
    },
  });
  pi.registerCommand("reregister-partial", {
    description: "Partially re-register dynamic without refreshModels",
    handler: async () => {
      pi.registerProvider("late-dynamic", {
        baseUrl: "https://late-v2.invalid/v1",
      });
    },
  });
}
`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	pigHome := t.TempDir()
	t.Setenv("PIG_HOME", pigHome)
	agentDir := filepath.Join(pigHome, "agent")

	services, err := CreateAgentSessionServices(CreateAgentSessionServicesOptions{CWD: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(services.Close)

	_, ext := setupNodeRefreshHost(t, services, extPath)
	if err := ext.Commands["set-agent-dir"].Handler(t.Context(), agentDir); err != nil {
		t.Fatal(err)
	}

	runtime := services.ModelRuntime()
	if runtime.GetModel("dynamic", "old") == nil {
		t.Fatal("initial model 'old' not registered")
	}

	// 1. Network refresh: publishes persistence, runs update synchronously, replaces models.
	result := runtime.Refresh(t.Context(), ai.ModelsRefreshOptions{
		Providers:    []string{"dynamic"},
		AllowNetwork: new(true),
		Force:        new(true),
	})
	if result.Aborted || len(result.Errors) != 0 {
		t.Fatalf("refresh failed: %+v", result)
	}
	if err := ext.Commands["check-calls"].Handler(t.Context(), ""); err != nil {
		t.Fatalf("check-calls: %v", err)
	}
	if err := ext.Commands["check-ctx"].Handler(t.Context(), `{"allowNetwork":true,"force":true,"baseUrl":"https://example.invalid/v1","hasCredential":true}`); err != nil {
		t.Fatalf("check-ctx: %v", err)
	}
	if got := runtime.GetModel("dynamic", "new"); got == nil || got.Capabilities.ContextWindow != 2000 {
		t.Fatalf("refreshed model 'new' = %+v; want ContextWindow=2000", got)
	} else {
		streamed := runtime.Complete(t.Context(), got, ai.Context{Messages: []ai.Message{ai.UserMessage{Content: ai.UserText("hi")}}}, ai.StreamOptions{})
		if streamed.StopReason != ai.StopReasonStop {
			t.Fatalf("streamed refreshed model = %+v", streamed)
		}
	}
	if got := runtime.GetModel("dynamic", "old"); got != nil {
		t.Fatalf("stale model 'old' still present: %+v", got)
	}

	// 2. Error propagation.
	if err := ext.Commands["set-flags"].Handler(t.Context(), `{"failNext":true}`); err != nil {
		t.Fatal(err)
	}
	errResult := runtime.Refresh(t.Context(), ai.ModelsRefreshOptions{
		Providers:    []string{"dynamic"},
		AllowNetwork: new(true),
	})
	if errResult.Errors["dynamic"] == nil || !strings.Contains(errResult.Errors["dynamic"].Error(), "refresh exploded") {
		t.Fatalf("expected error 'refresh exploded', got: %+v", errResult)
	}

	// 3. Cancellation via signal.
	if err := ext.Commands["set-flags"].Handler(t.Context(), `{"failNext":false,"waitSignal":true}`); err != nil {
		t.Fatal(err)
	}
	cancelCtx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	abortResult := runtime.Refresh(cancelCtx, ai.ModelsRefreshOptions{
		Providers:    []string{"dynamic"},
		AllowNetwork: new(true),
	})
	if !abortResult.Aborted {
		t.Fatalf("expected aborted refresh, got: %+v", abortResult)
	}
	if err := ext.Commands["set-flags"].Handler(t.Context(), `{"failNext":false,"waitSignal":false}`); err != nil {
		t.Fatal(err)
	}

	// 4. Late registration & partial re-registration keep refreshModels.
	if err := ext.Commands["late-register"].Handler(t.Context(), ""); err != nil {
		t.Fatal(err)
	}
	if err := ext.Commands["reregister-partial"].Handler(t.Context(), ""); err != nil {
		t.Fatal(err)
	}
	lateResult := runtime.Refresh(t.Context(), ai.ModelsRefreshOptions{
		Providers:    []string{"late-dynamic"},
		AllowNetwork: new(true),
	})
	if lateResult.Aborted || len(lateResult.Errors) != 0 {
		t.Fatalf("late refresh failed: %+v", lateResult)
	}
	if got := runtime.GetModel("late-dynamic", "late-new"); got == nil || got.Capabilities.ContextWindow != 4000 {
		t.Fatalf("late-new = %+v; want ContextWindow=4000", got)
	}

	// 5. Next process startup restores persisted models-store.json into ctx.stored on offline refresh.
	services2, err := CreateAgentSessionServices(CreateAgentSessionServicesOptions{CWD: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(services2.Close)
	setupNodeRefreshHost(t, services2, extPath)
	runtime2 := services2.ModelRuntime()
	offlineResult := runtime2.Refresh(t.Context(), ai.ModelsRefreshOptions{
		Providers:    []string{"dynamic"},
		AllowNetwork: new(false),
	})
	if offlineResult.Aborted || len(offlineResult.Errors) != 0 {
		t.Fatalf("offline refresh failed: %+v", offlineResult)
	}
	if got := runtime2.GetModel("dynamic", "restored"); got == nil || got.Capabilities.ContextWindow != 3000 {
		t.Fatalf("restored model = %+v; want ContextWindow=3000", got)
	}
}

func TestNodeProviderConfigRefreshModelsWithOAuthResolvesAndStreamsWithoutPanic(t *testing.T) {
	extDir := t.TempDir()
	extPath := filepath.Join(extDir, "oauth-refresh-provider.mjs")
	err := os.WriteFile(extPath, []byte(`
export default function (pi) {
  const ZERO = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  const usage = { ...ZERO, totalTokens: 0, cost: { ...ZERO, total: 0 } };
  pi.registerProvider("oauth-dynamic", {
    baseUrl: "https://example.invalid/v1",
    api: "oauth-dynamic-api",
    models: [
      { id: "old", name: "Old", input: ["text"], contextWindow: 1000, maxTokens: 100 },
    ],
    refreshModels: async () => [
      { id: "live", name: "Live", input: ["text"], contextWindow: 2000, maxTokens: 200 },
    ],
    oauth: {
      name: "OAuth Dynamic",
      isSubscription: true,
      login: async () => ({ access: "tok", refresh: "ref", expires: Date.now() + 3600_000 }),
      refreshToken: async (creds) => ({ ...creds, expires: Date.now() + 3600_000 }),
      getApiKey: (creds) => "oauth-key:" + creds.access,
    },
    streamSimple: (model, _ctx, options) => {
      const message = { role: "assistant", api: model.api, provider: model.provider, model: model.id, content: [{ type: "text", text: "key=" + options.apiKey }], usage, stopReason: "stop", timestamp: 1 };
      return { async *[Symbol.asyncIterator]() { yield { type: "start", partial: message }; yield { type: "done", reason: "stop", message }; }, result: async () => message };
    },
  });
}
`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	pigHome := t.TempDir()
	t.Setenv("PIG_HOME", pigHome)
	authDir := filepath.Join(pigHome, "agent")
	if err := os.MkdirAll(authDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(authDir, "auth.json"), []byte(`{"oauth-dynamic":{"type":"oauth","access":"acc-123","refresh":"ref-123","expires":9999999999999}}`), 0o600); err != nil {
		t.Fatal(err)
	}

	services, err := CreateAgentSessionServices(CreateAgentSessionServicesOptions{CWD: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(services.Close)
	setupNodeRefreshHost(t, services, extPath)

	runtime := services.ModelRuntime()
	res := runtime.Refresh(t.Context(), ai.ModelsRefreshOptions{
		Providers:    []string{"oauth-dynamic"},
		AllowNetwork: new(true),
	})
	if res.Aborted || len(res.Errors) != 0 {
		t.Fatalf("refresh failed: %+v", res)
	}
	model := runtime.GetModel("oauth-dynamic", "live")
	if model == nil {
		t.Fatal("refreshed model 'live' not found")
	}
	msg := runtime.Complete(t.Context(), model, ai.Context{Messages: []ai.Message{ai.UserMessage{Content: ai.UserText("hi")}}}, ai.StreamOptions{})
	if msg.StopReason != ai.StopReasonStop {
		t.Fatalf("complete failed: %+v (err=%q)", msg, msg.ErrorMessage)
	}
	if got := msg.Content[0].(ai.TextContent).Text; got != "key=oauth-key:acc-123" {
		t.Fatalf("got text %q, want %q", got, "key=oauth-key:acc-123")
	}
}
