import 'event_source_interface.dart';
import 'package:flutter/foundation.dart' show kDebugMode;


class MobileEventSourceService implements AppEventSource {
  @override
  void connect(String url, {
    required void Function() onOpen,
    required void Function(String data) onMessage,
    required void Function(String eventType, String data) onCustomEvent,
    required void Function() onError,
  }) {
    if (kDebugMode) {
      print("MobileEventSourceService: connect() called for $url. SSE not implemented for mobile in this stub.");
    }
    // Simulate immediate error or no-op for non-web platforms if SSE is web-only
    Future.microtask(onError); 
  }

  @override
  void close() {
    if (kDebugMode) {
      print("MobileEventSourceService: close() called.");
    }
  }
}
AppEventSource getEventSourceService() => MobileEventSourceService();