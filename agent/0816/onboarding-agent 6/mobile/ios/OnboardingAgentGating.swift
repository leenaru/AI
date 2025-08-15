import Foundation

struct LocalGateCfg { static let low = 0.45; static let high = 0.75; static let clipMax = 800 }
let REQUIRE_TOOLS: [String: Bool] = ["troubleshooting": true, "device_registration": true, "smalltalk": false]

struct IntentResult {
  let label: String
  let score: Double
  let slots: [String:String]
  let normalizedText: String?
}

enum Decision { case localAnswer(String); case clarify(String); case escalate(endpoint: String, payload: [String:Any]) }

// 온디바이스 LLM 엔진 인터페이스 (MLC/llama.cpp 등 바인딩으로 구현)
protocol LocalLLM { func runGemma(prompt: String) -> String }

protocol OnDeviceNLU { func inferIntent(_ text: String) -> IntentResult }

// Gemma 3n 프롬프트 -> JSON 파싱 NLU
final class GemmaNLU: OnDeviceNLU {
  private let engine: LocalLLM
  init(engine: LocalLLM) { self.engine = engine }

  func inferIntent(_ text: String) -> IntentResult {
    let prompt = """
      당신은 고객지원 NLU입니다. 아래 한국어 입력에서
      1) intent(label) ∈ {faq, troubleshooting, device_registration, smalltalk, other}
      2) score(0..1)
      3) slots: {error_code, product, severity, need_image}
      4) normalized_text
      5) safety: {pii:[], disallowed:false}
      를 JSON으로만 출력.

      입력:
      """ + text + """

      JSON:
    """
    let out = engine.runGemma(prompt: prompt)
    if let parsed = parseIntent(out) { return parsed }
    return ruleFallback(text)
  }

  private func parseIntent(_ json: String) -> IntentResult? {
    guard let data = json.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    let label = (obj["label"] as? String) ?? "faq"
    let score = (obj["score"] as? Double) ?? 0.55
    let slots = (obj["slots"] as? [String: Any])?.reduce(into: [String:String]()) { acc, kv in acc[kv.key] = String(describing: kv.value) } ?? [:]
    let nt = obj["normalized_text"] as? String
    return IntentResult(label: label, score: score, slots: slots, normalizedText: nt)
  }
}

// 파싱 실패 대비: 경량 룰 폴백
func ruleFallback(_ text: String) -> IntentResult {
  let t = text.lowercased()
  let label: String
  if t.contains("오류") || t.contains("에러") || t.contains("error") { label = "troubleshooting" }
  else if t.contains("등록") || t.contains("추가") { label = "device_registration" }
  else { label = "faq" }
  var slots: [String:String] = [:]
  if let r = text.range(of: #"[Ee][0-9]{2}"#, options: .regularExpression) { slots["error_code"] = String(text[r]) }
  if let r = text.range(of: #"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}"#, options: [.regularExpression, .caseInsensitive]) { slots["email"] = String(text[r]) }
  let score = (label == "troubleshooting") ? 0.78 : 0.55
  return IntentResult(label: label, score: score, slots: slots, normalizedText: nil)
}

enum PiiRedactor {
  static func redact(_ s: String) -> String {
    var out = s
    out = out.replacingOccurrences(of: #"[0-9]{3}-[0-9]{3,4}-[0-9]{4}"#, with: "[PHONE]", options: .regularExpression)
    out = out.replacingOccurrences(of: #"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}"#, with: "[EMAIL]", options: .regularExpression)
    return out
  }
}

enum PromptClipper {
  static func clip(_ text: String, slots: [String:String], max: Int = LocalGateCfg.clipMax) -> String {
    guard text.count > max else { return text }
    let keep = slots.values.joined(separator: " ")
    let head = String(text.prefix(max - keep.count - 10))
    return (head + " … " + keep).trimmingCharacters(in: .whitespaces)
  }
}

final class Gating {
  private let apiBase: String
  private let nlu: OnDeviceNLU
  init(apiBase: String, nlu: OnDeviceNLU) { self.apiBase = apiBase; self.nlu = nlu }

  func route(_ text: String) -> Decision {
    let res = nlu.inferIntent(text)
    let redacted = PiiRedactor.redact(text)
    let requiresTools = REQUIRE_TOOLS[res.label] ?? false

    if res.score < LocalGateCfg.low { return .clarify("질문을 조금 더 구체적으로 말씀해 주세요.") }
    if requiresTools || res.score < LocalGateCfg.high {
      let clipped = PromptClipper.clip(res.normalizedText ?? redacted, slots: res.slots)
      return .escalate(endpoint: apiBase + "/chat",
                       payload: [
                        "text": res.normalizedText ?? redacted,
                        "nlu": [
                          "label": res.label,
                          "score": res.score,
                          "slots": res.slots,
                          "normalized_text": res.normalizedText ?? redacted
                        ],
                        "device_ctx": ["platform":"ios"]
                       ])
    }
    return .localAnswer(localAnswerFor(res))
  }

  private func localAnswerFor(_ res: IntentResult) -> String {
    switch res.label {
    case "troubleshooting":
      let code = res.slots["error_code"].map { " (" + $0 + ")" } ?? ""
      return "다음 단계를 시도해 보세요: 1) 전원 재시작 2) 네트워크 확인 3) 오류코드 확인" + code
    case "device_registration":
      return "앱의 ‘기기 추가’에서 안내에 따라 진행해 주세요. 필요 시 카메라 버튼으로 QR을 스캔합니다."
    default:
      return "안내가 필요한 항목을 말씀해 주세요. (예: Wi-Fi 연결 방법)"
    }
  }

  func escalate(_ endpoint: String, payload: [String:Any], completion: @escaping (Bool, String)->Void) {
    guard let url = URL(string: endpoint),
          let data = try? JSONSerialization.data(withJSONObject: payload, options: []) else {
      completion(false, "bad request"); return
    }
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = data
    URLSession.shared.dataTask(with: req) { d, resp, err in
      if let err = err { completion(false, err.localizedDescription); return }
      let code = (resp as? HTTPURLResponse)?.statusCode ?? 500
      completion((200...299).contains(code), String(data: d ?? Data(), encoding: .utf8) ?? "")
    }.resume()
  }
}
