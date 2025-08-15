package ai.onboarding

/**
 * Android LocalLLM implementation backed by llama.cpp via JNI.
 * Loads a GGUF model (e.g., Gemma 3n) and returns full text from a streaming decode.
 *
 * NOTE:
 *  - You'll need to build the native library `libllama_bridge.so` (see CMakeLists.txt)
 *  - Ship or download the .gguf model to app-internal storage and pass its absolute path.
 */
class LlamaLocalLLM(
    modelPath: String,
    nCtx: Int = 2048,
    nThreads: Int = Runtime.getRuntime().availableProcessors().coerceAtMost(6)
) : LocalLLM {

    private var handle: Long = 0

    companion object {
        init {
            // Ensure your CMake target is named "llama_bridge"
            System.loadLibrary("llama_bridge")
        }
    }

    private external fun nativeInit(modelPath: String, nCtx: Int, nThreads: Int): Long
    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        repeatPenalty: Float,
        topK: Int,
        topP: Float,
        typicalP: Float
    ): String
    private external fun nativeFree(handle: Long)

    init {
        handle = nativeInit(modelPath, nCtx, nThreads)
        require(handle != 0L) { "Failed to init llama bridge (model path: $modelPath)" }
    }

    override fun runGemma(prompt: String): String {
        return nativeGenerate(
            handle = handle,
            prompt = prompt,
            maxTokens = 256,
            temperature = 0.8f,
            repeatPenalty = 1.1f,
            topK = 40,
            topP = 0.95f,
            typicalP = 1.0f
        )
    }

    fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }

    @Throws(Throwable::class)
    protected fun finalize() {
        close()
    }
}
