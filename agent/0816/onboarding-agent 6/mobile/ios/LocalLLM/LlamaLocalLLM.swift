//
//  LlamaLocalLLM.swift
//  OnboardingAgent iOS - LocalLLM (llama.cpp Swift binding sample)
//
import Foundation

// NOTE: This file assumes a Clang module exposing the C bridge header:
//   module CLlamaBridge { header "llama_bridge.h" }
// Then you can `import CLlamaBridge` below. For plain Xcode projects, add the header
// to a bridging header or a SwiftPM system library target.
import CLlamaBridge

// Conform to your existing protocol from OnboardingAgentGating.swift
public protocol LocalLLM { func runGemma(prompt: String) -> String }

// MARK: - Swift wrapper with streaming callback -> String accumulation
public final class LlamaLocalLLM: LocalLLM {
  private var ctx: OpaquePointer?

  public init(modelPath: String, nCtx: Int32 = 2048, nThreads: Int32 = 4) {
    self.ctx = modelPath.withCString { cpath in
      llama_bridge_init(cpath, nCtx, nThreads)
    }
  }

  deinit {
    if let ctx = ctx { llama_bridge_free(ctx) }
  }

  // Static callback thunk to receive tokens from C and append to a Swift buffer
  private final class StreamBox { var buffer = "" }

  private static let tokenThunk: (@convention(c) (UnsafePointer<CChar>?, Int32, UnsafeMutableRawPointer?) -> Void) = {
    (ptr, len, user) in
    guard let user = user else { return }
    let box = Unmanaged<StreamBox>.fromOpaque(user).takeUnretainedValue()
    if let ptr = ptr {
      // Unsafe bytes -> String (no copy)
      let s = String(bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr),
                     length: Int(len),
                     encoding: .utf8,
                     freeWhenDone: false) ?? ""
      box.buffer += s
    }
  }

  public func runGemma(prompt: String) -> String {
    guard let ctx = ctx else { return "" }
    let box = StreamBox()
    let ud = Unmanaged.passRetained(box).toOpaque()
    defer { Unmanaged<StreamBox>.fromOpaque(ud).release() }

    prompt.withCString { cPrompt in
      // Typical sampling params; tune as needed for your model
      llama_bridge_generate(
        ctx,
        cPrompt,
        LlamaLocalLLM.tokenThunk,
        ud,
        /*max_tokens*/ 256,
        /*temperature*/ 0.8,
        /*repeat_penalty*/ 1.1,
        /*top_k*/ 40,
        /*top_p*/ 0.95,
        /*typical_p*/ 1.0
      )
    }
    return box.buffer
  }
}
