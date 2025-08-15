package ai.onboarding

import android.util.Patterns
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

object LocalGateCfg {
    const val LOW = 0.45
    const val HIGH = 0.75
    val REQUIRE_TOOLS = mapOf("troubleshooting" to true, "device_registration" to true, "smalltalk" to false)
    const val CLIP_MAX_CHARS = 800
}
data class IntentResult(val label: String, val score: Double, val slots: Map<String,String> = emptyMap())
sealed class Decision {
    data class LocalAnswer(val text: String): Decision()
    data class Clarify(val prompt: String): Decision()
    data class Escalate(val endpoint: String, val payload: Map<String, Any?>): Decision()
}
interface OnDeviceNLU { fun inferIntent(text: String): IntentResult }
class RuleBasedNLU: OnDeviceNLU {
    private val errorKw = listOf("오류","에러","error","failed")
    private val photoKw = listOf("사진","이미지","첨부")
    override fun inferIntent(text: String): IntentResult {
        val t = text.lowercase()
        val label = when {
            errorKw.any { t.contains(it) } -> "troubleshooting"
            t.contains("등록") || t.contains("추가") -> "device_registration"
            else -> "faq"
        }
        val score = when {
            errorKw.any { t.contains(it) } -> 0.78
            photoKw.any { t.contains(it) } -> 0.70
            else -> 0.55
        }
        val slots = mutableMapOf<String,String>()
        Regex("[Ee]\d{2}").find(text)?.let { slots["error_code"] = it.value }
        Patterns.EMAIL_ADDRESS.matcher(text).takeIf { it.find() }?.let { slots["email"] = it.group() }
        Patterns.PHONE.matcher(text).takeIf { it.find() }?.let { slots["phone"] = it.group() }
        return IntentResult(label, score, slots)
    }
}
object PiiRedactor {
    fun redact(input: String): String {
        var out = input
        out = out.replace(Regex("\b\d{3}-\d{3,4}-\d{4}\b"), "[PHONE]")
        out = out.replace(Regex("\b010-?\d{4}-?\d{4}\b"), "[PHONE]")
        out = out.replace(Regex("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]")
        return out
    }
}
object PromptClipper {
    fun clip(text: String, slots: Map<String,String>, maxChars: Int = LocalGateCfg.CLIP_MAX_CHARS): String {
        if (text.length <= maxChars) return text
        val keep = slots.values.joinToString(" ")
        val head = text.take(maxChars - keep.length - 10)
        return "${head} … ${keep}".trim()
    }
}
class Gating(private val apiBase: String, private val http: OkHttpClient = OkHttpClient()) {
    private val nlu: OnDeviceNLU = RuleBasedNLU()
    fun route(text: String): Decision {
        val res = nlu.inferIntent(text)
        val redacted = PiiRedactor.redact(text)
        val requiresTools = LocalGateCfg.REQUIRE_TOOLS[res.label] == true
        return when {
            res.score < LocalGateCfg.LOW -> Decision.Clarify("질문을 조금 더 구체적으로 말씀해 주세요.")
            requiresTools || res.score < LocalGateCfg.HIGH -> {
                val clipped = PromptClipper.clip(redacted, res.slots)
                Decision.Escalate(
                    endpoint = "$apiBase/chat",
                    payload = mapOf("text" to clipped, "intent" to res.label, "score" to res.score,
                                    "slots" to res.slots, "device_ctx" to mapOf("platform" to "android"))
                )
            }
            else -> Decision.LocalAnswer(localAnswerFor(res))
        }
    }
    private fun localAnswerFor(res: IntentResult): String = when (res.label) {
        "troubleshooting" -> "다음 단계를 시도해 보세요: 1) 전원 재시작 2) 네트워크 확인 3) 오류코드 확인" + (res.slots["error_code"]?.let { " ($it)" } ?: "")
        "device_registration" -> "앱의 ‘기기 추가’에서 안내에 따라 진행해 주세요. 필요 시 카메라 버튼으로 QR을 스캔합니다."
        else -> "안내가 필요한 항목을 말씀해 주세요. (예: Wi-Fi 연결 방법)"
    }
    fun escalate(decision: Decision.Escalate, onResult: (Boolean, String) -> Unit) {
        val media = "application/json; charset=utf-8".toMediaType()
        val json = JSONObject(decision.payload).toString()
        val req = Request.Builder().url(decision.endpoint).post(json.toRequestBody(media)).build()
        CoroutineScope(Dispatchers.IO).launch {
            try {
                http.newCall(req).execute().use { resp ->
                    val ok = resp.isSuccessful
                    val body = resp.body?.string() ?: ""
                    onResult(ok, body)
                }
            } catch (e: Exception) { onResult(false, e.message ?: "error") }
        }
    }
}
