// llama_bridge.h - Minimal C bridge around llama.cpp for Swift
// This is a SAMPLE header; implemented in llama_bridge.c and linked with llama.cpp.
// Copyright: OSI-friendly
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context
typedef void* llama_bridge_ctx;

// Streaming callback signature: UTF-8 token chunk
typedef void (*llama_token_callback)(const char* token, int32_t len, void* user_data);

// Initialize a model context from GGUF file
llama_bridge_ctx llama_bridge_init(const char* model_path, int32_t n_ctx, int32_t n_threads);

// Optional: set system prompt (can be no-op if your template embeds it)
void llama_bridge_set_system(llama_bridge_ctx ctx, const char* system_utf8);

// Generate with streaming callback; returns 0 on success
int32_t llama_bridge_generate(
    llama_bridge_ctx ctx,
    const char* prompt_utf8,
    llama_token_callback cb,
    void* user_data,
    int32_t max_tokens,
    float temperature,
    float repeat_penalty,
    int32_t top_k,
    float top_p,
    float typical_p
);

// Free resources
void llama_bridge_free(llama_bridge_ctx ctx);

#ifdef __cplusplus
}
#endif
