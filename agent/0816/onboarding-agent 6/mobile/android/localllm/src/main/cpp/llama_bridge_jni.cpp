#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include <android/log.h>

#define LOG_TAG "LlamaBridgeJNI"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

#ifndef HAS_LLAMA
// Demo-only mode: no llama.cpp linked. Returns canned JSON so you can wire end-to-end.
struct demo_ctx { int dummy; };
extern "C" JNIEXPORT jlong JNICALL
Java_ai_onboarding_LlamaLocalLLM_nativeInit(JNIEnv* env, jobject thiz, jstring jpath, jint nCtx, jint nThreads) {
    (void)env; (void)thiz; (void)jpath; (void)nCtx; (void)nThreads;
    demo_ctx* ctx = new demo_ctx{1};
    return reinterpret_cast<jlong>(ctx);
}
extern "C" JNIEXPORT jstring JNICALL
Java_ai_onboarding_LlamaLocalLLM_nativeGenerate(JNIEnv* env, jobject thiz, jlong handle, jstring jprompt, jint maxTokens,
                                                jfloat temperature, jfloat repeatPenalty, jint topK, jfloat topP, jfloat typicalP) {
    (void)thiz; (void)handle; (void)jprompt; (void)maxTokens; (void)temperature; (void)repeatPenalty; (void)topK; (void)topP; (void)typicalP;
    const char* demo = "{\"label\":\"faq\",\"score\":0.62,\"slots\":{},\"normalized_text\":\"요청 요약\"}";
    return env->NewStringUTF(demo);
}
extern "C" JNIEXPORT void JNICALL
Java_ai_onboarding_LlamaLocalLLM_nativeFree(JNIEnv* env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    auto* ctx = reinterpret_cast<demo_ctx*>(handle);
    delete ctx;
}
#else
// Real llama.cpp-backed implementation
#include "llama.h"

struct real_ctx {
    llama_model*   model;
    llama_context* lctx;
    int32_t n_ctx;
    int32_t n_threads;
};

static int tokenize_prompt(llama_model* model, const char* text, std::vector<llama_token>& out) {
    const int32_t cap = (int32_t)strlen(text) + 8;
    out.resize(cap);
    const int32_t n = llama_tokenize(model, text, (int32_t)strlen(text), out.data(), cap, /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) return -1;
    out.resize(n);
    return n;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onboarding_LlamaLocalLLM_nativeInit(JNIEnv* env, jobject thiz, jstring jpath, jint nCtx, jint nThreads) {
    (void)thiz;
    const char* cpath = env->GetStringUTFChars(jpath, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 33; // tune per device; 0 = CPU
    mparams.main_gpu = 0;

    llama_model* model = llama_load_model_from_file(cpath, mparams);
    env->ReleaseStringUTFChars(jpath, cpath);
    if (!model) {
        ALOGE("Failed to load GGUF model");
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx > 0 ? nCtx : 2048;
    cparams.n_threads = nThreads > 0 ? nThreads : 4;

    llama_context* lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) {
        ALOGE("Failed to create llama context");
        llama_free_model(model);
        return 0;
    }

    auto* ctx = new real_ctx{model, lctx, cparams.n_ctx, cparams.n_threads};
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_onboarding_LlamaLocalLLM_nativeGenerate(JNIEnv* env, jobject thiz, jlong handle, jstring jprompt, jint maxTokens,
                                                jfloat temperature, jfloat repeatPenalty, jint topK, jfloat topP, jfloat typicalP) {
    (void)thiz;
    auto* r = reinterpret_cast<real_ctx*>(handle);
    if (!r) return env->NewStringUTF("");

    const char* cprompt = env->GetStringUTFChars(jprompt, nullptr);

    // Tokenize
    std::vector<llama_token> prompt_toks;
    if (tokenize_prompt(r->model, cprompt, prompt_toks) < 0) {
        env->ReleaseStringUTFChars(jprompt, cprompt);
        return env->NewStringUTF("");
    }

    // Evaluate prompt in chunks
    int32_t n_past = 0;
    llama_batch batch = llama_batch_init(r->n_ctx, 0, 1);

    int i = 0;
    while (i < (int)prompt_toks.size()) {
        const int n_eval = std::min((int)prompt_toks.size() - i, (int)r->n_ctx);
        llama_batch_clear(&batch);
        for (int j = 0; j < n_eval; ++j) {
            llama_batch_add(&batch, prompt_toks[i + j], n_past + j, NULL, 0);
        }
        if (llama_decode(r->lctx, batch) != 0) {
            ALOGE("llama_decode(prompt) failed");
            llama_batch_free(batch);
            env->ReleaseStringUTFChars(jprompt, cprompt);
            return env->NewStringUTF("");
        }
        n_past += n_eval;
        i += n_eval;
    }

    env->ReleaseStringUTFChars(jprompt, cprompt);

    // Sampling loop
    const int32_t n_vocab = llama_n_vocab(r->model);
    std::vector<llama_token_data> data(n_vocab);
    const int repeat_last_n = 64;
    std::vector<llama_token> last;
    last.reserve(repeat_last_n);

    std::string out;
    out.reserve(4096);

    int produced = 0;
    while (produced < maxTokens) {
        const float* logits = llama_get_logits(r->lctx);
        for (int t = 0; t < n_vocab; ++t) {
            data[t].id = t;
            data[t].logit = logits[t];
            data[t].p = 0.0f;
        }
        llama_token_data_array candidates = { data.data(), (size_t)n_vocab, false };

        if (!last.empty()) {
            llama_sample_repetition_penalty(r->lctx, &candidates, last.data(), (int)last.size(), repeatPenalty);
        }
        if (topK > 0)         llama_sample_top_k(r->lctx, &candidates, topK, 1);
        if (topP < 1.0f)      llama_sample_top_p(r->lctx, &candidates, topP, 1);
        if (typicalP < 1.0f)  llama_sample_typical(r->lctx, &candidates, typicalP, 1);
        if (temperature != 1.0f) llama_sample_temperature(r->lctx, &candidates, temperature);

        const llama_token tok = llama_sample_token(r->lctx, &candidates);

        char piece[512];
        const int n = llama_token_to_piece(r->model, tok, piece, (int)sizeof(piece), /*special*/ false, /*normalize*/ true);
        if (n > 0) out.append(piece, (size_t)n);

        if (tok == llama_token_eos(r->model)) break;

        if ((int)last.size() < repeat_last_n) last.push_back(tok);
        else {
            memmove(last.data(), last.data() + 1, sizeof(llama_token) * (repeat_last_n - 1));
            last[repeat_last_n - 1] = tok;
        }

        llama_batch_clear(&batch);
        llama_batch_add(&batch, tok, n_past, NULL, 0);
        if (llama_decode(r->lctx, batch) != 0) {
            ALOGE("llama_decode(step) failed");
            break;
        }
        n_past += 1;
        produced += 1;
    }

    llama_batch_free(batch);

    // Return UTF-8 (JNI expects Modified UTF-8; for practical ASCII/UTF-8 tokens it's fine)
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onboarding_LlamaLocalLLM_nativeFree(JNIEnv* env, jobject thiz, jlong handle) {
    (void)env; (void)thiz;
    auto* r = reinterpret_cast<real_ctx*>(handle);
    if (!r) return;
    if (r->lctx) llama_free(r->lctx);
    if (r->model) llama_free_model(r->model);
    llama_backend_free();
    delete r;
}
#endif
