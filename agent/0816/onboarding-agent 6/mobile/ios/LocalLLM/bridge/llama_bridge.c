// llama_bridge.c - Minimal C bridge around llama.cpp for Swift
#include "llama_bridge.h"
#include <string.h>
#include <stdlib.h>

#ifndef HAS_LLAMA

// -------- Demo stub (no llama.cpp linked) --------
struct demo_ctx { int dummy; };

llama_bridge_ctx llama_bridge_init(const char* model_path, int32_t n_ctx, int32_t n_threads) {
    (void)model_path; (void)n_ctx; (void)n_threads;
    struct demo_ctx* ctx = (struct demo_ctx*)malloc(sizeof(struct demo_ctx));
    ctx->dummy = 1;
    return (llama_bridge_ctx)ctx;
}

void llama_bridge_set_system(llama_bridge_ctx ctx, const char* system_utf8) {
    (void)ctx; (void)system_utf8;
}

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
){
    (void)ctx; (void)prompt_utf8; (void)max_tokens; (void)temperature; (void)repeat_penalty;
    (void)top_k; (void)top_p; (void)typical_p;
    const char* demo = "{\"label\":\"faq\",\"score\":0.62,\"slots\":{},\"normalized_text\":\"요청 요약\"}";
    cb(demo, (int32_t)strlen(demo), user_data);
    return 0;
}

void llama_bridge_free(llama_bridge_ctx ctx) {
    if (ctx) free(ctx);
}

#else

// -------- Real implementation (requires llama.cpp) --------
#include "llama.h"   // from llama.cpp
#include <stdio.h>

struct real_ctx {
    struct llama_model*   model;
    struct llama_context* lctx;
    int32_t n_ctx;
    int32_t n_threads;
};

llama_bridge_ctx llama_bridge_init(const char* model_path, int32_t n_ctx, int32_t n_threads) {
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    // Tune for iOS/Metal: let llama.cpp use GPU layers if available
    mparams.n_gpu_layers = 33; // adjust per device; 0 = CPU only
    mparams.main_gpu     = 0;

    struct llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[llama_bridge] failed to load model: %s\n", model_path);
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx > 0 ? n_ctx : 2048;
    cparams.seed      = 0;
    cparams.n_threads = n_threads > 0 ? n_threads : 4;

    struct llama_context* lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) {
        fprintf(stderr, "[llama_bridge] failed to create context\n");
        llama_free_model(model);
        return NULL;
    }

    struct real_ctx* r = (struct real_ctx*)malloc(sizeof(struct real_ctx));
    r->model = model;
    r->lctx  = lctx;
    r->n_ctx = cparams.n_ctx;
    r->n_threads = cparams.n_threads;
    return (llama_bridge_ctx)r;
}

void llama_bridge_set_system(llama_bridge_ctx ctx, const char* system_utf8) {
    // Optional: keep for future use, not needed for simple JSON NLU prompt
    (void)ctx; (void)system_utf8;
}

// Helper: tokenize UTF-8 string into llama tokens (adds BOS)
static int tokenize_prompt(struct llama_model* model, const char* text, llama_token** out, int32_t* out_n) {
    const int32_t max = (int32_t)strlen(text) + 8; // generous upper bound
    llama_token* toks = (llama_token*)malloc(sizeof(llama_token) * max);
    if (!toks) return -1;
    const int32_t n = llama_tokenize(model, text, (int32_t)strlen(text), toks, max, /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) { free(toks); return -1; }
    *out = toks; *out_n = n;
    return 0;
}

int32_t llama_bridge_generate(
    llama_bridge_ctx handle,
    const char* prompt_utf8,
    llama_token_callback cb,
    void* user_data,
    int32_t max_tokens,
    float temperature,
    float repeat_penalty,
    int32_t top_k,
    float top_p,
    float typical_p
){
    if (!handle || !prompt_utf8 || !cb) return -1;
    struct real_ctx* r = (struct real_ctx*)handle;
    struct llama_context* ctx = r->lctx;
    struct llama_model*   model = r->model;

    // 1) Tokenize prompt
    llama_token* prompt_toks = NULL;
    int32_t n_prompt = 0;
    if (tokenize_prompt(model, prompt_utf8, &prompt_toks, &n_prompt) != 0) return -2;

    // 2) Eval prompt in chunks (respect n_ctx)
    int32_t n_past = 0;
    llama_batch batch = llama_batch_init(r->n_ctx, 0, 1);

    int32_t i = 0;
    while (i < n_prompt) {
        const int32_t n_eval = (n_prompt - i > r->n_ctx) ? r->n_ctx : (n_prompt - i);
        llama_batch_clear(&batch);
        for (int32_t j = 0; j < n_eval; ++j) {
            llama_batch_add(&batch, prompt_toks[i + j], n_past + j, NULL, 0);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[llama_bridge] llama_decode(prompt) failed\n");
            llama_batch_free(batch);
            free(prompt_toks);
            return -3;
        }
        n_past += n_eval;
        i += n_eval;
    }
    free(prompt_toks);

    // 3) Sampling loop
    const int32_t n_vocab = llama_n_vocab(model);
    llama_token_data* data = (llama_token_data*)malloc(sizeof(llama_token_data) * (size_t)n_vocab);
    if (!data) { llama_batch_free(batch); return -4; }

    // repetition buffer
    const int32_t repeat_last_n = 64;
    llama_token last_tokens[repeat_last_n];
    int32_t last_count = 0;

    int32_t produced = 0;
    while (produced < max_tokens) {
        // logits of the last token
        const float* logits = llama_get_logits(ctx);
        for (int32_t t = 0; t < n_vocab; ++t) {
            data[t].id = t;
            data[t].logit = logits[t];
            data[t].p = 0.0f;
        }
        llama_token_data_array candidates = { data, (size_t)n_vocab, false };

        // sampling filters
        if (repeat_last_n > 0 && last_count > 0) {
            llama_sample_repetition_penalty(ctx, &candidates, last_tokens, last_count, repeat_penalty);
        }
        if (top_k > 0)      llama_sample_top_k(ctx, &candidates, top_k, 1);
        if (top_p < 1.0f)   llama_sample_top_p(ctx, &candidates, top_p, 1);
        if (typical_p < 1.0f) llama_sample_typical(ctx, &candidates, typical_p, 1);
        if (temperature != 1.0f) llama_sample_temperature(ctx, &candidates, temperature);

        const llama_token tok = llama_sample_token(ctx, &candidates);

        // detokenize piece and stream out
        char piece[512];
        const int n = llama_token_to_piece(model, tok, piece, (int)sizeof(piece), /*special*/ false, /*normalize*/ true);
        if (n > 0) cb(piece, (int32_t)n, user_data);

        // stop conditions
        if (tok == llama_token_eos(model)) break;

        // append to history ring buffer
        if (last_count < repeat_last_n) {
            last_tokens[last_count++] = tok;
        } else {
            memmove(last_tokens, last_tokens + 1, sizeof(llama_token) * (repeat_last_n - 1));
            last_tokens[repeat_last_n - 1] = tok;
        }

        // feed token for next step
        llama_batch_clear(&batch);
        llama_batch_add(&batch, tok, n_past, NULL, 0);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[llama_bridge] llama_decode(step) failed\n");
            break;
        }
        n_past += 1;
        produced += 1;
    }

    free(data);
    llama_batch_free(batch);
    return 0;
}

void llama_bridge_free(llama_bridge_ctx handle) {
    if (!handle) return;
    struct real_ctx* r = (struct real_ctx*)handle;
    if (r->lctx)  llama_free(r->lctx);
    if (r->model) llama_free_model(r->model);
    llama_backend_free();
    free(r);
}

#endif
