# Android integration example

```kotlin
// Somewhere in your Application/DI setup:
val modelPath = File(filesDir, "models/gemma-3n-Q4_K_M.gguf").absolutePath
val engine: LocalLLM = LlamaLocalLLM(modelPath = modelPath, nCtx = 2048, nThreads = 4)
val nlu: OnDeviceNLU = GemmaNLU(engine)
val gate = Gating(apiBase = "https://api.example.com", nlu = nlu)

// Later, per user text:
when (val d = gate.route(userText)) {
    is Decision.LocalAnswer -> showText(d.text)
    is Decision.Clarify -> showClarify(d.prompt)
    is Decision.Escalate -> gate.escalate(d) { ok, body -> /* handle */ }
}
```

**Model file**: Do not bundle large GGUF inside the APK. Download on first run (e.g., WorkManager) into `filesDir/models/` and verify SHA256 before use.
