import Foundation

struct LocalGateCfg { static let low = 0.45; static let high = 0.75 }

class OnDeviceLLM {
  func inferIntent(_ text: String) -> (String, Double, [String:String]) {
    let label = (text.contains("오류") || text.contains("에러")) ? "troubleshooting" : "faq"
    let score = (text.contains("사진") || text.contains("이미지")) ? 0.7 : 0.5
    return (label, score, [:])
  }
}

enum Action {
  case clarify(String)
  case escalate(String, [String:Any])
  case localAnswer(String)
}

class Gating {
  let llm = OnDeviceLLM()
  func route(_ text: String) -> Action {
    let (label, score, slots) = llm.inferIntent(text)
    if score < LocalGateCfg.low { return .clarify("질문을 조금 더 구체적으로 말씀해 주세요") }
    if score < LocalGateCfg.high { return .escalate("/chat", ["text": text, "intent": label, "slots": slots]) }
    return .localAnswer("간단 답변: \(label)")
  }
}
