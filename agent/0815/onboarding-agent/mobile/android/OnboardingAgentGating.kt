package ai.onboarding

object LocalGateCfg {
    const val LOW = 0.45
    const val HIGH = 0.75
}

class OnDeviceLLM {
    fun inferIntent(text: String): Triple<String, Double, Map<String, String>> {
        val label = if (text.contains("오류") || text.contains("에러")) "troubleshooting" else "faq"
        val score = if (text.contains("사진") || text.contains("이미지")) 0.7 else 0.5
        return Triple(label, score, emptyMap())
    }
}

class Gating {
    private val llm = OnDeviceLLM()
    fun route(text: String): Action {
        val (label, score, slots) = llm.inferIntent(text)
        return when {
            score < LocalGateCfg.LOW -> Action.Clarify("질문을 조금 더 구체적으로 말씀해 주세요")
            score < LocalGateCfg.HIGH -> Action.Escalate("/chat", mapOf("text" to text, "intent" to label, "slots" to slots))
            else -> Action.LocalAnswer("간단 답변: $label")
        }
    }
}

sealed class Action {
    data class Clarify(val msg: String): Action()
    data class Escalate(val path: String, val payload: Map<String, Any?>): Action()
    data class LocalAnswer(val msg: String): Action()
}
