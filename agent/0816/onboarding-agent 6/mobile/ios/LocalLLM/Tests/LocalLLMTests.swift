import XCTest
@testable import LocalLLM

final class LocalLLMTests: XCTestCase {
  func testEngineInit() {
    let engine = LlamaLocalLLM(modelPath: "/dev/null", nCtx: 512, nThreads: 1)
    _ = engine.runGemma(prompt: "hello")
  }
}
