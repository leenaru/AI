# WorkManager-based model download (Android)

## Enqueue
```kotlin
val url = "https://example.cdn/models/gemma-3n-Q4_K_M.gguf"
val sha256 = "0123456789abcdef..." // expected SHA-256 (hex)
val id = ModelDownloader.enqueue(appContext, url, sha256)
// Observe status:
WorkManager.getInstance(appContext).getWorkInfoByIdLiveData(id).observe(this) { info ->
    if (info != null && info.state.isFinished) {
        val path = info.outputData.getString("path")
        // Construct engine when ready
        if (path != null) {
            val engine: LocalLLM = LlamaLocalLLM(modelPath = path, nCtx = 2048, nThreads = 4)
            // ...
        }
    }
}
```

## Why foreground
- Long-running downloads need a foreground notification on modern Android.
- This worker shows progress and verifies **SHA-256** before atomic move.

## Notes
- Store under `filesDir/models/` to avoid requiring `READ_EXTERNAL_STORAGE`.
- Use **unmetered** network and backoff to protect user data/battery.
