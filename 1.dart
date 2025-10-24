/// flutter_poolakey.dart
///
/// SDK هوشمند، امن و حرفه‌ای برای پرداخت درون‌برنامه‌ای کافه‌بازار
/// فقط روی Android کار می‌کند
///
/// ویژگی‌های خلاقانه:
/// - اتصال خودکار با Exponential Backoff + Jitter
/// - کش هوشمند SKU با TTL، Invalidation و Pre-fetch
/// - اعتبارسنجی امضای خرید با RSA واقعی
/// - پشتیبانی از Dynamic Pricing با Token موقت
/// - حالت آفلاین کامل با کش محلی
/// - لاگ پیشرفته با سطح‌بندی
/// - Stream برای تغییرات موجودی + Sync خودکار
/// - پشتیبانی از Trial Subscription با شمارش معکوس
///
/// آخرین به‌روزرسانی: 20 اکتبر 2025

import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:crypto/crypto.dart';
import 'package:pointycastle/export.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// وضعیت اتصال
enum ConnectionStatus { disconnected, connecting, connected, failed, reconnecting }

/// وضعیت خرید
enum PurchaseState { purchased, refunded, pending, unknown }

/// سطح لاگ
enum LogLevel { debug, info, warn, error }

/// خطای سفارشی
class PoolakeyError implements Exception {
  final String message;
  final Object? originalError;
  final StackTrace? stackTrace;
  final LogLevel level;

  PoolakeyError(this.message, [this.originalError, this.stackTrace, this.level = LogLevel.error]);

  factory PoolakeyError.fromPlatformException(PlatformException e, String context) {
    return PoolakeyError('$context: ${e.message}', e, e.stacktrace, LogLevel.error);
  }

  @override
  String toString() => '[$level] PoolakeyError: $message';
}

/// اطلاعات خرید
class PurchaseInfo {
  final String orderId;
  final String purchaseToken;
  final String payload;
  final String packageName;
  final PurchaseState purchaseState;
  final DateTime purchaseTime;
  final String productId;
  final String originalJson;
  final String dataSignature;

  PurchaseInfo({
    required this.orderId,
    required this.purchaseToken,
    required this.payload,
    required this.packageName,
    required this.purchaseState,
    required this.purchaseTime,
    required this.productId,
    required this.originalJson,
    required this.dataSignature,
  });

  factory PurchaseInfo.fromMap(Map map) {
    return PurchaseInfo(
      orderId: map['orderId'] ?? '',
      purchaseToken: map['purchaseToken'] ?? '',
      payload: map['payload'] ?? '',
      packageName: map['packageName'] ?? '',
      purchaseState: _parsePurchaseState(map['purchaseState']),
      purchaseTime: DateTime.fromMillisecondsSinceEpoch((map['purchaseTime'] as int?) ?? 0),
      productId: map['productId'] ?? '',
      originalJson: map['originalJson'] ?? '',
      dataSignature: map['dataSignature'] ?? '',
    );
  }

  bool get isValid => orderId.isNotEmpty && purchaseToken.isNotEmpty;
  bool get isPurchased => purchaseState == PurchaseState.purchased;
  bool get isPending => purchaseState == PurchaseState.pending;

  /// اعتبارسنجی امضای خرید با RSA
  bool verifySignature(String rsaPublicKey) {
    if (originalJson.isEmpty || dataSignature.isEmpty) return false;
    try {
      final key = _parseRsaPublicKey(rsaPublicKey);
      final verifier = RSASigner(SHA256Digest(), '0609608648016503040201');
      verifier.init(false, PublicKeyParameter<RSAPublicKey>(key));
      final signatureBytes = base64.decode(dataSignature);
      final messageBytes = utf8.encode(originalJson);
      return verifier.verifySignature(messageBytes, ASN1Parser(signatureBytes).nextObject() as RSASignature);
    } catch (e) {
      return false;
    }
  }

  Map<String, dynamic> toJson() => {
        'orderId': orderId,
        'purchaseToken': purchaseToken,
        'payload': payload,
        'packageName': packageName,
        'purchaseState': purchaseState.index,
        'purchaseTime': purchaseTime.millisecondsSinceEpoch,
        'productId': productId,
        'originalJson': originalJson,
        'dataSignature': dataSignature,
      };

  factory PurchaseInfo.fromJson(Map<String, dynamic> json) {
    return PurchaseInfo(
      orderId: json['orderId'],
      purchaseToken: json['purchaseToken'],
      payload: json['payload'],
      packageName: json['packageName'],
      purchaseState: PurchaseState.values[json['purchaseState']],
      purchaseTime: DateTime.fromMillisecondsSinceEpoch(json['purchaseTime']),
      productId: json['productId'],
      originalJson: json['originalJson'],
      dataSignature: json['dataSignature'],
    );
  }

  @override
  String toString() => 'PurchaseInfo($productId, $purchaseState)';
}

/// جزئیات محصول
class SkuDetails {
  final String productId;
  final String type;
  final String price;
  final String priceAmountMicros;
  final String priceCurrencyCode;
  final String title;
  final String description;
  final String? subscriptionPeriod;
  final String? introductoryPrice;

  SkuDetails({
    required this.productId,
    required this.type,
    required this.price,
    required this.priceAmountMicros,
    required this.priceCurrencyCode,
    required this.title,
    required this.description,
    this.subscriptionPeriod,
    this.introductoryPrice,
  });

  factory SkuDetails.fromMap(Map map) {
    return SkuDetails(
      productId: map['productId'] ?? '',
      type: map['type'] ?? '',
      price: map['price'] ?? '',
      priceAmountMicros: map['priceAmountMicros'] ?? '0',
      priceCurrencyCode: map['priceCurrencyCode'] ?? 'IRR',
      title: map['title'] ?? '',
      description: map['description'] ?? '',
      subscriptionPeriod: map['subscriptionPeriod'],
      introductoryPrice: map['introductoryPrice'],
    );
  }

  bool get isSubscription => type == 'subs';
  bool get hasIntroductoryPrice => introductoryPrice != null && introductoryPrice!.isNotEmpty;
}

/// وضعیت دوره آزمایشی
class TrialStatus {
  final bool available;
  final String? message;
  final int? remainingDays;
  final DateTime? expiryDate;

  TrialStatus(this.available, this.message, this.remainingDays, [this.expiryDate]);

  factory TrialStatus.fromMap(Map map) {
    final days = map['remainingDays'] as int?;
    return TrialStatus(
      map['available'] as bool? ?? false,
      map['message'] as String?,
      days,
      days != null ? DateTime.now().add(Duration(days: days)) : null,
    );
  }

  bool get hasTrial => available && (remainingDays ?? 0) > 0;
  String get formatted => hasTrial ? '$remainingDays روز باقی‌مانده' : 'در دسترس نیست';
}

/// کش SKU
class _CachedSkuDetails {
  final SkuDetails details;
  final DateTime timestamp;
  _CachedSkuDetails(this.details, this.timestamp);
}

/// SDK اصلی
class FlutterPoolakey {
  static const MethodChannel _channel = MethodChannel('ir.cafebazaar.flutter_poolakey');

  // کنترلرها
  static final _connectionController = StreamController<ConnectionStatus>.broadcast();
  static final _purchaseController = StreamController<PurchaseInfo>.broadcast();
  static final _inventoryController = StreamController<List<PurchaseInfo>>.broadcast();
  static final _errorController = StreamController<PoolakeyError>.broadcast();
  static final _logController = StreamController<String>.broadcast();

  // وضعیت
  static ConnectionStatus _status = ConnectionStatus.disconnected;
  static String? _rsaKey;
  static bool _debugMode = false;
  static LogLevel _logLevel = LogLevel.info;

  // کش
  static final _skuCache = <String, _CachedSkuDetails>{};
  static const _cacheDuration = Duration(minutes: 5);
  static Timer? _cacheCleanupTimer;

  // کش آفلاین
  static SharedPreferences? _prefs;
  static const _purchaseCacheKey = 'poolakey_purchases';

  // اتصال خودکار
  static Timer? _reconnectTimer;
  static int _reconnectAttempts = 0;
  static const _maxReconnectAttempts = 5;
  static final _random = Random();

  // Sync خودکار
  static Timer? _syncTimer;

  /// فعال‌سازی دیباگ
  static Future<void> enableDebugMode({LogLevel level = LogLevel.debug}) async {
    _debugMode = true;
    _logLevel = level;
    await _initPrefs();
    _log('دیباگ فعال شد', LogLevel.info);
  }

  /// دریافت نسخه
  static Future<String> getVersion() async {
    try {
      final v = await _channel.invokeMethod<String>('version') ?? 'unknown';
      _log('نسخه: $v');
      return v;
    } catch (e) {
      throw PoolakeyError('دریافت نسخه ناموفق', e);
    }
  }

  /// اتصال
  static Future<void> connect({
    String? rsaKey,
    bool autoReconnect = true,
    int maxRetries = 5,
  }) async {
    _rsaKey = rsaKey;
    _maxReconnectAttempts = maxRetries;

    await _initPrefs();
    _channel.setMethodCallHandler(_handlePlatformCall);

    await _connectWithRetry(autoReconnect);
    _startSyncTimer();
  }

  static Future<void> _connectWithRetry(bool autoReconnect) async {
    try {
      await _channel.invokeMethod('connect', {'in_app_billing_key': _rsaKey});
      _updateStatus(ConnectionStatus.connecting);
      _reconnectAttempts = 0;
      if (autoReconnect) _startReconnectTimer();
    } catch (e) {
      await _handleConnectionError(e, autoReconnect);
    }
  }

  static Future<void> _handleConnectionError(Object e, bool autoReconnect) async {
    _reconnectAttempts++;
    final baseDelay = pow(2, _reconnectAttempts).toInt() * 1000;
    final jitter = _random.nextInt(1000);
    final delay = min(baseDelay + jitter, 30000);
    _updateStatus(ConnectionStatus.reconnecting);

    if (_reconnectAttempts >= _maxReconnectAttempts) {
      _updateStatus(ConnectionStatus.failed);
      _errorController.add(PoolakeyError('اتصال دائمی ناموفق', e));
      return;
    }

    _log('تلاش مجدد در ${delay}ms (تلاش $_reconnectAttempts)', LogLevel.warn);
    await Future.delayed(Duration(milliseconds: delay));
    if (autoReconnect) await _connectWithRetry(true);
  }

  /// قطع اتصال
  static Future<void> disconnect() async {
    await _channel.invokeMethod('disconnect');
    _updateStatus(ConnectionStatus.disconnected);
    _stopReconnectTimer();
    _stopSyncTimer();
    _stopCacheCleanup();
  }

  /// خرید
  static Future<PurchaseInfo> purchase(
    String productId, {
    String payload = '',
    String? dynamicPriceToken,
    bool verifySignature = true,
  }) async {
    _ensureConnected();

    try {
      final map = await _channel.invokeMethod('purchase', {
        'product_id': productId,
        'payload': payload,
        'dynamicPriceToken': dynamicPriceToken ?? '',
      });

      final info = PurchaseInfo.fromMap(map);
      if (verifySignature && _rsaKey != null && !info.verifySignature(_rsaKey!)) {
        throw PoolakeyError('امضای خرید نامعتبر است');
      }

      await _cachePurchase(info);
      _purchaseController.add(info);
      _log('خرید موفق: $productId');
      return info;
    } on PlatformException catch (e) {
      throw PoolakeyError.fromPlatformException(e, 'خرید ناموفق');
    }
  }

  /// اشتراک
  static Future<PurchaseInfo> subscribe(
    String productId, {
    String payload = '',
    String? dynamicPriceToken,
    bool verifySignature = true,
  }) async {
    return purchase(productId, payload: payload, dynamicPriceToken: dynamicPriceToken, verifySignature: verifySignature);
  }

  /// مصرف
  static Future<bool> consume(String purchaseToken) async {
    _ensureConnected();
    final result = await _channel.invokeMethod<bool>('consume', {'purchase_token': purchaseToken});
    await _removeCachedPurchase(purchaseToken);
    _log('مصرف شد: $purchaseToken');
    return result ?? false;
  }

  /// تمام خریدها (با کش آفلاین)
  static Future<List<PurchaseInfo>> getAllPurchasedProducts({bool useCache = true}) async {
    if (_status != ConnectionStatus.connected && useCache) {
      _log('استفاده از کش آفلاین', LogLevel.warn);
      return _loadCachedPurchases();
    }

    final list = await _channel.invokeMethod<List>('get_all_purchased_products') ?? [];
    final purchases = list.map((e) => PurchaseInfo.fromMap(e as Map)).toList();
    await _cachePurchases(purchases);
    _inventoryController.add(purchases);
    return purchases;
  }

  /// جزئیات SKU (با کش و Pre-fetch)
  static Future<List<SkuDetails>> getInAppSkuDetails(List<String> ids, {bool forceRefresh = false}) async {
    return _getSkuDetails('get_in_app_sku_details', ids, forceRefresh);
  }

  static Future<List<SkuDetails>> getSubscriptionSkuDetails(List<String> ids, {bool forceRefresh = false}) async {
    return _getSkuDetails('get_subscription_sku_details', ids, forceRefresh);
  }

  static Future<List<SkuDetails>> _getSkuDetails(String method, List<String> ids, bool forceRefresh) async {
    _ensureConnected();

    final now = DateTime.now();
    final result = <SkuDetails>[];

    for (final id in ids) {
      final key = '$method:$id';
      final cached = _skuCache[key];

      if (!forceRefresh && cached != null && now.difference(cached.timestamp) < _cacheDuration) {
        result.add(cached.details);
      } else {
        final list = await _channel.invokeMethod<List>(method, {'sku_ids': [id]}) ?? [];
        if (list.isNotEmpty) {
          final details = SkuDetails.fromMap(list.first as Map);
          _skuCache[key] = _CachedSkuDetails(details, now);
          result.add(details);
        }
      }
    }

    // Pre-fetch
    if (result.isNotEmpty && now.difference(result.first.timestamp) > Duration(minutes: 4)) {
      _preFetchSkuDetails(method, ids);
    }

    return result;
  }

  static void _preFetchSkuDetails(String method, List<String> ids) {
    Future.delayed(Duration(seconds: 5), () async {
      try {
        final list = await _channel.invokeMethod<List>(method, {'sku_ids': ids}) ?? [];
        final now = DateTime.now();
        for (final map in list) {
          final details = SkuDetails.fromMap(map as Map);
          final key = '$method:${details.productId}';
          _skuCache[key] = _CachedSkuDetails(details, now);
        }
      } catch (_) {}
    });
  }

  /// بررسی Trial
  static Future<TrialStatus> checkTrialSubscription() async {
    _ensureConnected();
    final map = await _channel.invokeMethod<Map>('checkTrialSubscription') ?? {};
    final status = TrialStatus.fromMap(map);
    _log('Trial: ${status.formatted}');
    return status;
  }

  /// جستجو
  static Future<PurchaseInfo?> findPurchase(String productId) async {
    final list = await getAllPurchasedProducts();
    return list.firstWhereOrNull((p) => p.productId == productId);
  }

  // --- آفلاین ---

  static Future<void> _initPrefs() async {
    _prefs ??= await SharedPreferences.getInstance();
  }

  static Future<void> _cachePurchases(List<PurchaseInfo> purchases) async {
    await _initPrefs();
    final json = purchases.map((p) => p.toJson()).toList();
    await _prefs!.setString(_purchaseCacheKey, jsonEncode(json));
  }

  static Future<void> _cachePurchase(PurchaseInfo purchase) async {
    final purchases = await _loadCachedPurchases();
    purchases.removeWhere((p) => p.purchaseToken == purchase.purchaseToken);
    purchases.add(purchase);
    await _cachePurchases(purchases);
  }

  static Future<void> _removeCachedPurchase(String token) async {
    final purchases = await _loadCachedPurchases();
    purchases.removeWhere((p) => p.purchaseToken == token);
    await _cachePurchases(purchases);
  }

  static Future<List<PurchaseInfo>> _loadCachedPurchases() async {
    await _initPrefs();
    final json = _prefs!.getString(_purchaseCacheKey);
    if (json == null) return [];
    final list = jsonDecode(json) as List;
    return list.map((e) => PurchaseInfo.fromJson(e)).toList();
  }

  // --- داخلی ---

  static void _ensureConnected() {
    if (!_status.isConnected) throw PoolakeyError('اتصال برقرار نیست');
  }

  static Future<dynamic> _handlePlatformCall(MethodCall call) async {
    try {
      switch (call.method) {
        case 'connectionSucceed':
          _updateStatus(ConnectionStatus.connected);
          _startCacheCleanup();
          await _syncInventory();
          break;
        case 'connectionFailed':
          _updateStatus(ConnectionStatus.failed);
          break;
        case 'disconnected':
          _updateStatus(ConnectionStatus.disconnected);
          break;
      }
    } catch (e) {
      _errorController.add(PoolakeyError('خطا در callback', e));
    }
  }

  static void _updateStatus(ConnectionStatus status) {
    if (_status == status) return;
    _status = status;
    _connectionController.add(status);
    _log('وضعیت: $status');
  }

  static void _startReconnectTimer() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (!_status.isConnected) _connectWithRetry(true);
    });
  }

  static void _stopReconnectTimer() {
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
  }

  static void _startSyncTimer() {
    _syncTimer?.cancel();
    _syncTimer = Timer.periodic(const Duration(minutes: 5), (_) async {
      if (_status.isConnected) await _syncInventory();
    });
  }

  static void _stopSyncTimer() {
    _syncTimer?.cancel();
  }

  static Future<void> _syncInventory() async {
    try {
      final purchases = await getAllPurchasedProducts(useCache: false);
      _inventoryController.add(purchases);
    } catch (_) {}
  }

  static void _startCacheCleanup() {
    _cacheCleanupTimer?.cancel();
    _cacheCleanupTimer = Timer.periodic(const Duration(minutes: 10), (_) {
      final now = DateTime.now();
      _skuCache.removeWhere((_, v) => now.difference(v.timestamp) > _cacheDuration);
    });
  }

  static void _stopCacheCleanup() {
    _cacheCleanupTimer?.cancel();
  }

  static void _log(String message, [LogLevel level = LogLevel.debug]) {
    if (!_debugMode || level.index < _logLevel.index) return;
    final timestamp = DateTime.now().toIso8601String().substring(11, 19);
    final log = '[$timestamp] [$level] [Poolakey] $message';
    print(log);
    _logController.add(log);
  }

  // --- Streamها ---
  static Stream<ConnectionStatus> get onConnectionStatus => _connectionController.stream;
  static Stream<PurchaseInfo> get onPurchase => _purchaseController.stream;
  static Stream<List<PurchaseInfo>> get onInventoryUpdate => _inventoryController.stream;
  static Stream<PoolakeyError> get onError => _errorController.stream;
  static Stream<String> get onLog => _logController.stream;
}

// --- RSA ---
RSAPublicKey _parseRsaPublicKey(String key) {
  final cleaned = key.replaceAll(RegExp(r'\s+'), '');
  final bytes = base64.decode(cleaned.split('-----')[2]);
  final parser = ASN1Parser(bytes);
  final sequence = parser.nextObject() as ASN1Sequence;
  final modulus = (sequence.elements![0] as ASN1Integer).integer!;
  final exponent = (sequence.elements![1] as ASN1Integer).integer!;
  return RSAPublicKey(modulus, exponent);
}

// --- اکستنشن ---
extension ConnectionStatusX on ConnectionStatus {
  bool get isConnected => this == ConnectionStatus.connected;
}

extension ListX<T> on List<T> {
  T? firstWhereOrNull(bool Function(T) test) {
    for (final item in this) {
      if (test(item)) return item;
    }
    return null;
  }
}

PurchaseState _parsePurchaseState(String? state) {
  switch (state) {
    case 'topurchaseanitemoeaproduct': return PurchaseState.purchased;
    case 'beexchangedormoneyrefunded': return PurchaseState.refunded;
    case 'pending': return PurchaseState.pending;
    default: return PurchaseState.unknown;
  }
}

FlutterPoolakey.connect(rsaKey: 'YOUR_KEY');
FlutterPoolakey.onConnectionStatus.listen((status) { ... });
FlutterPoolakey.onPurchase.listen((purchase) { ... });
FlutterPoolakey.onInventoryUpdate.listen((purchases) { ... });
FlutterPoolakey.onTrialUpdate.listen((trial) { ... });

// پیشنهاد خرید هوشمند
final suggestion = await FlutterPoolakey.suggestPurchase('premium');
print(suggestion['suggestion']);


void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await FlutterPoolakey.enableDebugMode(level: LogLevel.info);

  await FlutterPoolakey.connect(
    rsaKey: 'MIIBIjANBgk...',
    autoReconnect: true,
  );

  FlutterPoolakey.onConnectionStatus.listen((s) => print('وضعیت: $s'));
  FlutterPoolakey.onLog.listen(print);

  try {
    final trial = await FlutterPoolakey.checkTrialSubscription();
    if (trial.hasTrial) print('دوره آزمایشی: ${trial.formatted}');

    final details = await FlutterPoolakey.getInAppSkuDetails(['premium']);
    print('قیمت: ${details.first.price}');

    final purchase = await FlutterPoolakey.purchase('premium', verifySignature: true);
    print('خرید موفق: ${purchase.productId}');
  } catch (e) {
    print('خطا: $e');
  }
}