package ai.onboarding.localllm.work

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * Enqueue a unique model download with Wi-Fi/unmetered constraint and exponential backoff.
 */
object ModelDownloader {
    fun enqueue(context: Context, url: String, sha256: String, filename: String = "gemma-3n-Q4_K_M.gguf"): UUID {
        val input = Data.Builder()
            .putString(ModelDownloadWorker.KEY_URL, url)
            .putString(ModelDownloadWorker.KEY_SHA256, sha256)
            .putString(ModelDownloadWorker.KEY_FILENAME, filename)
            .build()

        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.UNMETERED)
            .setRequiresBatteryNotLow(true)
            .setRequiresStorageNotLow(true)
            .build()

        val req = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .setInputData(input)
            .addTag("model-download")
            .build()

        // Use unique name per SHA256 so we don't duplicate work
        val name = "model-$sha256"
        WorkManager.getInstance(context).enqueueUniqueWork(name, ExistingWorkPolicy.KEEP, req)
        return req.id
    }
}
