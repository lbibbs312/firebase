abstract class AppEventSource {
  void connect(String url, {
    required void Function() onOpen,
    required void Function(String data) onMessage, // For unnamed messages
    required void Function(String eventType, String data) onCustomEvent, // For named events
    required void Function() onError,
  });
  void close();
  // bool get isOpen; // This was tricky with dart:html, can be omitted or re-thought
}