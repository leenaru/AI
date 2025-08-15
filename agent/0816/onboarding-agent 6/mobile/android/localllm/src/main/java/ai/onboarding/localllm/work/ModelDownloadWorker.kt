package ai.onboarding.localllm.work

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okio.buffer
import okio.sink
import java.io.File
import java.security.DigestInputStream
import java.security.MessageDigest

/**
 * Downloads a GGUF model with resume support, verifies SHA-256, and atomically moves into place.
 * Foreground notification is used for reliability.
 */
class ModelDownloadWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {

    companion object {
        const val KEY_URL = "url"
        const val KEY_SHA256 = "sha256"
        const val KEY_FILENAME = "filename"
        const val CHANNEL_ID = "model_dl"
        const val NOTIF_ID = 4201
    }

    private val client = OkHttpClient.Builder()
        .retryOnConnectionFailure(true)
        .build()

    override suspend fun getForegroundInfo(): ForegroundInfo {
        return createForegroundInfo(0, 0L, 0L, "대기 중…")
    }

    private fun createForegroundInfo(progress: Int, read: Long, total: Long, text: String): ForegroundInfo {
        val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(CHANNEL_ID, "Model Download", NotificationManager.IMPORTANCE_LOW)
            nm.createNotificationChannel(ch)
        }
        val notif: Notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("모델 다운로드")
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setProgress(100, progress, total <= 0)
            .build()
        return ForegroundInfo(NOTIF_ID, notif)
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val url = inputData.getString(KEY_URL) ?: return@withContext Result.failure(Data.Builder().putString("error", "missing url").build())
        val sha256 = inputData.getString(KEY_SHA256) ?: return@withContext Result.failure(Data.Builder().putString("error", "missing sha256").build())
        val filename = inputData.getString(KEY_FILENAME) ?: "gemma-3n-Q4_K_M.gguf"

        val modelsDir = File(applicationContext.filesDir, "models").apply { mkdirs() }
        val dest = File(modelsDir, filename)
        val part = File(modelsDir, "$filename.part")

        // If already present and hash matches, succeed fast
        if (dest.exists()) {
            val ok = HashUtils.verifySha256(dest, sha256)
            if (ok) return@withContext Result.success(Data.Builder().putString("path", dest.absolutePath).build())
            dest.delete()
        }

        var downloaded = if (part.exists()) part.length() else 0L
        var totalBytes = -1L

        setForeground(createForegroundInfo(0, downloaded, totalBytes, "다운로드 시작…"))

        val reqBuilder = Request.Builder().url(url)
        if (downloaded > 0) reqBuilder.addHeader("Range", "bytes=$downloaded-")
        val req = reqBuilder.build()

        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                return@withContext Result.retry()
            }
            totalBytes = try { resp.headers["Content-Length"]?.toLong() ?: -1L } catch (_: Throwable) { -1L }
            val body = resp.body ?: return@withContext Result.retry()

            // If server ignored Range, start over
            if (resp.code == 200 && downloaded > 0) {
                downloaded = 0L
                part.delete()
            }

            part.parentFile?.mkdirs()
            part.createNewFile()
            body.source().use { source ->
                part.sink(append = true).buffer().use { sink ->
                    val bufferSize = 512 * 1024
                    var bytesRead: Long
                    var lastNotified = System.currentTimeMillis()
                    while (true) {
                        bytesRead = source.read(sink.buffer, bufferSize.toLong())
                        if (bytesRead == -1L) break
                        sink.emit()
                        downloaded += bytesRead
                        val now = System.currentTimeMillis()
                        if (now - lastNotified > 500) {
                            val progress = if (totalBytes > 0) ((downloaded * 100) / (downloaded + (totalBytes))) else 0
                            setForeground(createForegroundInfo(progress.toInt(), downloaded, totalBytes, "다운로드 중…"))
                            lastNotified = now
                        }
                    }
                }
            }
        }

        // Verify SHA256
        setForeground(createForegroundInfo(100, downloaded, totalBytes, "무결성 검증…"))
        if (!HashUtils.verifySha256(part, sha256)) {
            part.delete()
            return@withContext Result.retry()
        }

        // Atomic move into place
        if (dest.exists()) dest.delete()
        if (!part.renameTo(dest)) {
            return@withContext Result.retry()
        }

        Result.success(Data.Builder().putString("path", dest.absolutePath).build())
    }
}

object HashUtils {
    fun verifySha256(file: File, expectedHex: String): Boolean {
        val hex = sha256Hex(file)
        return expectedHex.equals(hex, ignoreCase = true)
    }

    fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { fis ->
            val buf = ByteArray(1 shl 20) // 1MB
            while (true) {
                val r = fis.read(buf)
                if (r < 0) break
                digest.update(buf, 0, r)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
