package ir.cafebazaar.flutter_poolakey

import android.app.Activity
import android.content.Context
import android.content.Intent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.NonNull
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.Result
import ir.cafebazaar.poolakey.Payment
import ir.cafebazaar.poolakey.callback.ConnectionCallback
import ir.cafebazaar.poolakey.callback.PurchaseCallback
import ir.cafebazaar.poolakey.config.PaymentConfiguration
import ir.cafebazaar.poolakey.config.SecurityCheck
import ir.cafebazaar.poolakey.entity.PurchaseInfo
import ir.cafebazaar.poolakey.entity.SkuDetails
import ir.cafebazaar.poolakey.request.PurchaseRequest
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import timber.log.Timber
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.pow

/**
 * FlutterPoolakeyPlugin
 * SDK هوشمند، امن و حرفه‌ای برای پرداخت درون‌برنامه‌ای کافه‌بازار
 * آخرین به‌روزرسانی: 20 اکتبر 2025
 */
class FlutterPoolakeyPlugin : FlutterPlugin, ActivityAware, MethodChannel.MethodCallHandler {

    private var binding: ActivityPluginBinding? = null
    private var channel: MethodChannel? = null
    private var context: Context? = null
    private var activity: Activity? = null

    private var payment: Payment? = null
    private var paymentConnection: ir.cafebazaar.poolakey.Connection? = null
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // --- Streamها ---
    private val _connectionState = MutableLiveData<ConnectionState>()
    val connectionState: LiveData<ConnectionState> get() = _connectionState

    private val _purchaseFlow = MutableStateFlow<PurchaseInfo?>(null)
    val purchaseFlow: StateFlow<PurchaseInfo?> get() = _purchaseFlow

    private val _inventoryFlow = MutableStateFlow<List<PurchaseInfo>>(emptyList())
    val inventoryFlow: StateFlow<List<PurchaseInfo>> get() = _inventoryFlow

    private val _errorFlow = MutableStateFlow<String?>(null)
    val errorFlow: StateFlow<String?> get() = _errorFlow

    private val _trialFlow = MutableStateFlow<TrialStatus?>(null)
    val trialFlow: StateFlow<TrialStatus?> get() = _trialFlow

    // --- کش ---
    private val skuCache = CacheManager<SkuDetails>()
    private val purchaseCache = CacheManager<PurchaseInfo>()
    private val dynamicTokenCache = CacheManager<String>()

    // --- اتصال ---
    private val reconnectAttempts = AtomicInteger(0)
    private var reconnectJob: Job? = null
    private var circuitBreakerOpen = false

    // --- خرید ---
    private var purchaseLauncher: ActivityResultLauncher<PurchaseRequest>? = null
    private val pendingResult = AtomicReference<Result?>(null)

    override fun onAttachedToEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        this.context = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, "ir.cafebazaar.flutter_poolakey")
        channel?.setMethodCallHandler(this)
        setupTimber()
        CacheManager.initGson()
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        this.binding = binding
        this.activity = binding.activity
        setupPurchaseLauncher()
    }

    override fun onDetachedFromActivityForConfigChanges() = clearActivity()
    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) = onAttachedToActivity(binding)
    override fun onDetachedFromActivity() = clearActivity()

    private fun clearActivity() {
        binding = null
        activity = null
        purchaseLauncher = null
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel?.setMethodCallHandler(null)
        scope.cancel()
        disconnect()
    }

    // --- MethodChannel ---
    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "version" -> result.success("1.5.0")
            "connect" -> connect(call.argument("in_app_billing_key"), result)
            "disconnect" -> disconnect(result)
            "purchase" -> purchase(call, result, false)
            "subscribe" -> purchase(call, result, true)
            "consume" -> consume(call.argument("purchase_token")!!, result)
            "get_all_purchased_products" -> getPurchasedProducts(result, false)
            "get_all_subscribed_products" -> getPurchasedProducts(result, true)
            "get_in_app_sku_details" -> getSkuDetails(call.argument("sku_ids")!!, result, false)
            "get_subscription_sku_details" -> getSkuDetails(call.argument("sku_ids")!!, result, true)
            "checkTrialSubscription" -> checkTrialSubscription(result)
            "suggestPurchase" -> suggestPurchase(call.argument("product_id")!!, result)
            else -> result.notImplemented()
        }
    }

    // --- اتصال هوشمند ---
    private fun connect(rsaKey: String?, result: Result) {
        if (circuitBreakerOpen) {
            result.error("CIRCUIT_BREAKER", "Too many failures. Try later.", null)
            return
        }

        val securityCheck = rsaKey?.let { SecurityCheck.Enable(it) } ?: SecurityCheck.Disable
        val config = PaymentConfiguration(localSecurityCheck = securityCheck)

        payment = Payment(context!!, config)
        paymentConnection = payment!!.connect(createConnectionCallback())
        result.success(null)
    }

    private fun createConnectionCallback(): ConnectionCallback.() -> Unit = {
        connectionSucceed {
            _connectionState.value = ConnectionState.Connected
            channel?.invokeMethod("connectionSucceed", null)
            reconnectAttempts.set(0)
            circuitBreakerOpen = false
            startAutoSync()
            loadCachedPurchases()
            preFetchSkuDetails()
        }
        connectionFailed {
            _connectionState.value = ConnectionState.Failed
            _errorFlow.value = it.message
            channel?.invokeMethod("connectionFailed", it.message)
            scheduleReconnect()
        }
        disconnected {
            _connectionState.value = ConnectionState.Disconnected
            channel?.invokeMethod("disconnected", null)
            scheduleReconnect()
        }
    }

    private fun scheduleReconnect() {
        reconnectJob?.cancel()
        val attempts = reconnectAttempts.incrementAndGet()
        if (attempts > 5) {
            circuitBreakerOpen = true
            scope.launch {
                delay(30 * 60 * 1000) // 30 دقیقه
                circuitBreakerOpen = false
                reconnectAttempts.set(0)
            }
            return
        }

        val baseDelay = 2.0.pow(attempts - 1).toLong() * 1000L
        val jitter = (0..1000).random()
        val delay = baseDelay + jitter

        reconnectJob = scope.launch {
            delay(delay)
            paymentConnection = payment?.connect(createConnectionCallback())
        }
    }

    private fun disconnect(result: Result? = null) {
        paymentConnection?.disconnect()
        reconnectJob?.cancel()
        reconnectAttempts.set(0)
        circuitBreakerOpen = false
        scope.cancel()
        result?.success(null)
    }

    // --- خرید هوشمند ---
    private fun setupPurchaseLauncher() {
        purchaseLauncher = binding?.activity?.registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { activityResult ->
            val result = pendingResult.getAndSet(null) ?: return@registerForActivityResult
            if (activityResult.resultCode == Activity.RESULT_OK) {
                val purchase = activityResult.data?.getParcelableExtra<PurchaseInfo>("purchase")
                purchase?.let {
                    cachePurchase(it)
                    _purchaseFlow.value = it
                    result.success(it.toMap())
                } ?: result.error("PURCHASE_FAILED", "No purchase data", null)
            } else {
                result.error("PURCHASE_CANCELLED", "User cancelled", null)
            }
        }
    }

    private fun purchase(call: MethodCall, result: Result, isSubscription: Boolean) {
        ensureConnected(result) {
            val productId = call.argument<String>("product_id")!!
            val payload = call.argument<String>("payload") ?: ""
            val dynamicToken = call.argument<String>("dynamicPriceToken")

            // استفاده از توکن موقت
            val token = dynamicToken ?: dynamicTokenCache.get("dynamic:$productId")
            val request = PurchaseRequest(productId, payload, token)
            pendingResult.set(result)

            val callback = createPurchaseCallback()
            if (isSubscription) {
                payment?.subscribeProduct(purchaseLauncher!!, request, callback)
            } else {
                payment?.purchaseProduct(purchaseLauncher!!, request, callback)
            }
        }
    }

    private fun createPurchaseCallback(): PurchaseCallback.() -> Unit = {
        purchaseSucceed {
            cachePurchase(it)
            _purchaseFlow.value = it
            pendingResult.get()?.success(it.toMap())
            pendingResult.set(null)
        }
        purchaseCanceled {
            pendingResult.get()?.error("PURCHASE_CANCELLED", "User cancelled", null)
            pendingResult.set(null)
        }
        purchaseFailed {
            pendingResult.get()?.error("PURCHASE_FAILED", it.message, null)
            pendingResult.set(null)
        }
    }

    // --- مصرف ---
    private fun consume(token: String, result: Result) {
        ensureConnected(result) {
            payment?.consumeProduct(token) {
                consumeSucceed {
                    removeFromCache(token)
                    result.success(true)
                }
                consumeFailed { result.error("CONSUME_FAILED", it.message, null) }
            }
        }
    }

    // --- SKU با کش + Pre-fetch ---
    private fun getSkuDetails(skuIds: List<String>, result: Result, isSubscription: Boolean) {
        ensureConnected(result) {
            val type = if (isSubscription) "sub" else "inapp"
            val cached = skuIds.mapNotNull { skuCache.get("$type:$it") }
            val uncached = skuIds.filter { skuCache.get("$type:$it") == null }

            if (uncached.isEmpty()) {
                result.success(cached.map { it.toMap() })
                return@ensureConnected
            }

            val method = if (isSubscription) payment!!::getSubscriptionSkuDetails else payment!!::getInAppSkuDetails
            method.invoke(uncached) {
                getSkuDetailsSucceed {
                    it.forEach { sku -> skuCache.put("$type:${sku.sku}", sku) }
                    result.success((cached + it).map { it.toMap() })
                }
                getSkuDetailsFailed { result.error("SKU_FAILED", it.message, null) }
            }
        }
    }

    private fun preFetchSkuDetails() {
        scope.launch {
            delay(5000)
            val popular = listOf("premium", "pro", "basic")
            getSkuDetails(popular, SilentResult(), false)
            getSkuDetails(popular, SilentResult(), true)
        }
    }

    // --- خریدهای آفلاین ---
    private fun getPurchasedProducts(result: Result, isSubscription: Boolean) {
        ensureConnected(result) {
            val method = if (isSubscription) payment!!::getSubscribedProducts else payment!!::getPurchasedProducts
            method.invoke {
                querySucceed {
                    it.forEach { p -> cachePurchase(p) }
                    _inventoryFlow.value = it
                    result.success(it.map { p -> p.toMap() })
                }
                queryFailed {
                    val cached = if (isSubscription) purchaseCache.values.filter { it.isSubscription } else purchaseCache.values.filter { !it.isSubscription }
                    if (cached.isNotEmpty()) {
                        result.success(cached.map { it.toMap() })
                    } else {
                        result.error("QUERY_FAILED", it.message, null)
                    }
                }
            }
        }
    }

    private fun checkTrialSubscription(result: Result) {
        ensureConnected(result) {
            payment?.checkTrialSubscription {
                checkTrialSubscriptionSucceed {
                    val status = TrialStatus(it.isAvailable, it.trialPeriodDays, it.message)
                    _trialFlow.value = status
                    result.success(status.toMap())
                }
                checkTrialSubscriptionFailed { result.error("TRIAL_FAILED", it.message, null) }
            }
        }
    }

    // --- پیشنهاد خرید هوشمند ---
    private fun suggestPurchase(productId: String, result: Result) {
        val lastPurchase = purchaseCache.values.maxByOrNull { it.purchaseTime }
        val suggestion = when {
            lastPurchase == null -> "خرید $productId توصیه می‌شود"
            lastPurchase.purchaseTime < System.currentTimeMillis() - 30L * 24 * 60 * 60 * 1000 -> "مدت زیادی از آخرین خرید گذشته"
            else -> "شما اخیراً خرید کرده‌اید"
        }
        result.success(mapOf("suggestion" to suggestion, "productId" to productId))
    }

    // --- کش ---
    private fun cachePurchase(info: PurchaseInfo) {
        purchaseCache.put(info.purchaseToken, info)
        CacheManager.saveToDisk(context!!, "purchases", purchaseCache.values.toList())
    }

    private fun removeFromCache(token: String) {
        purchaseCache.remove(token)
        CacheManager.saveToDisk(context!!, "purchases", purchaseCache.values.toList())
    }

    private fun loadCachedPurchases() {
        val cached = CacheManager.loadFromDisk<List<PurchaseInfo>>(context!!, "purchases") ?: return
        purchaseCache.putAll(cached.associateBy { it.purchaseToken })
        _inventoryFlow.value = cached
    }

    // --- همگام‌سازی خودکار ---
    private fun startAutoSync() {
        scope.launch {
            while (isActive) {
                delay(5 * 60 * 1000)
                if (_connectionState.value == ConnectionState.Connected) {
                    syncInventory()
                }
            }
        }
    }

    private fun syncInventory() {
        getPurchasedProducts(SilentResult(), false)
        getPurchasedProducts(SilentResult(), true)
    }

    // --- ابزارها ---
    private fun ensureConnected(result: Result, block: () -> Unit) {
        if (_connectionState.value != ConnectionState.Connected) {
            result.error("NOT_CONNECTED", "Poolakey is not connected", null)
        } else {
            block()
        }
    }

    private fun setupTimber() {
        Timber.plant(object : Timber.DebugTree() {
            override fun createStackElementTag(element: StackTraceElement): String = "Poolakey"
        })
        if (BuildConfig.DEBUG) {
            Timber.plant(FileLoggingTree(context!!))
        }
    }
}

// --- کش هوشمند ---
class CacheManager<T : Any> {
    private val memoryCache = mutableMapOf<String, CacheEntry<T>>()
    private val ttl = 5 * 60 * 1000L // 5 دقیقه

    companion object {
        lateinit var gson: Gson
        fun initGson() { gson = Gson() }
    }

    fun put(key: String, value: T) {
        memoryCache[key] = CacheEntry(value, System.currentTimeMillis())
    }

    fun get(key: String): T? {
        val entry = memoryCache[key] ?: return null
        if (System.currentTimeMillis() - entry.timestamp > ttl) {
            memoryCache.remove(key)
            return null
        }
        return entry.value
    }

    fun remove(key: String) = memoryCache.remove(key)
    fun putAll(items: Map<String, T>) = items.forEach { (k, v) -> put(k, v) }
    fun values() = memoryCache.values.map { it.value }
    fun isNotEmpty() = memoryCache.isNotEmpty()

    data class CacheEntry<T>(val value: T, val timestamp: Long)

    companion object {
        fun <T> saveToDisk(context: Context, key: String, data: List<T>) {
            val file = File(context.filesDir, "$key.json")
            file.writeText(gson.toJson(data))
        }

        inline fun <reified T> loadFromDisk(context: Context, key: String): List<T>? {
            val file = File(context.filesDir, "$key.json")
            if (!file.exists()) return null
            return gson.fromJson(file.readText(), object : TypeToken<List<T>>() {}.type)
        }
    }
}

// --- وضعیت Trial ---
data class TrialStatus(
    val available: Boolean,
    val remainingDays: Int,
    val message: String?
) {
    fun toMap() = mapOf(
        "available" to available,
        "remainingDays" to remainingDays,
        "message" to message,
        "formatted" to if (available && remainingDays > 0) "$remainingDays روز باقی‌مانده" else "در دسترس نیست"
    )
}

// --- لاگ‌گیری ---
class FileLoggingTree(context: Context) : Timber.Tree() {
    private val logFile = File(context.filesDir, "poolakey.log")
    private val formatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
        val time = formatter.format(Date())
        logFile.appendText("[$time] [$tag] $message${t?.let { "\n${it.stackTraceToString()}" } ?: ""}\n")
    }
}

// --- تبدیل ایمن ---
internal fun PurchaseInfo.toMap() = mapOf(
    "orderId" to orderId,
    "purchaseToken" to purchaseToken,
    "payload" to payload,
    "packageName" to packageName,
    "purchaseState" to when (purchaseState) {
        ir.cafebazaar.poolakey.entity.PurchaseState.PURCHASED -> "purchased"
        ir.cafebazaar.poolakey.entity.PurchaseState.REFUNDED -> "refunded"
        ir.cafebazaar.poolakey.entity.PurchaseState.PENDING -> "pending"
        else -> "unknown"
    },
    "purchaseTime" to purchaseTime,
    "productId" to productId,
    "originalJson" to originalJson,
    "dataSignature" to dataSignature,
    "isSubscription" to (subscriptionPeriod != null)
)

internal fun SkuDetails.toMap() = mapOf(
    "productId" to sku,
    "type" to type,
    "price" to price,
    "priceAmountMicros" to priceAmountMicros,
    "priceCurrencyCode" to priceCurrencyCode,
    "title" to title,
    "description" to description,
    "subscriptionPeriod" to subscriptionPeriod,
    "introductoryPrice" to introductoryPrice
)

// --- وضعیت اتصال ---
enum class ConnectionState { Disconnected, Connecting, Connected, Failed }

// --- Silent Result ---
class SilentResult : Result {
    override fun success(result: Any?) {}
    override fun error(code: String, message: String?, details: Any?) {}
    override fun notImplemented() {}
}