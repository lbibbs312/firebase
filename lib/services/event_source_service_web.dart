import 'dart:html' as html;
import 'event_source_interface.dart';

class HtmlEventSourceService implements AppEventSource {
  html.EventSource? _eventSource;
  // Store listeners to remove them on close, or re-add on reconnect
  Map<String, html.EventListener> _customEventListeners = {};
  html.EventListener? _onMessageListener;
  html.EventListener? _onOpenListener;
  html.EventListener? _onErrorListener;

  @override
  void connect(String url, {
    required void Function() onOpen,
    required void Function(String data) onMessage,
    required void Function(String eventType, String data) onCustomEvent,
    required void Function() onError,
  }) {
    close(); // Close any existing connection
    _eventSource = html.EventSource(url);

    _onOpenListener = (html.Event event) => onOpen();
    _eventSource!.onOpen.listen(_onOpenListener);

    _onMessageListener = (html.Event event) {
      if (event is html.MessageEvent) {
        onMessage(event.data as String);
      }
    };
    _eventSource!.onMessage.listen(_onMessageListener);
    
    // Define a common set of custom events you expect
    final List<String> expectedCustomEvents = [
      'ask_user', 'progress', 'agent_speech', 
      'agent_thought_event', 'agent_plan_event', 'agent_criticism_event', 
      'agent_tool_event', 'web_search_progress_event', 
      'sse_stream_step_completed' 
      // Add any other named events your backend might send
    ];

    _customEventListeners = {}; // Clear previous listeners
    for (var eventName in expectedCustomEvents) {
      final listener = (html.Event event) {
        if (event is html.MessageEvent) {
          onCustomEvent(eventName, event.data as String);
        }
      };
      _eventSource!.addEventListener(eventName, listener);
      _customEventListeners[eventName] = listener;
    }

    _onErrorListener = (html.Event event) => onError();
    _eventSource!.onError.listen(_onErrorListener);
  }

  @override
  void close() {
    if (_eventSource != null) {
      // Remove specific listeners if they were stored
      if (_onOpenListener != null) _eventSource!.onOpen.listen(null); // This is how you remove dart:html listeners
      if (_onMessageListener != null) _eventSource!.onMessage.listen(null);
      if (_onErrorListener != null) _eventSource!.onError.listen(null);
      _customEventListeners.forEach((name, listener) {
        _eventSource!.removeEventListener(name, listener);
      });
      _customEventListeners.clear();

      _eventSource!.close();
      _eventSource = null;
    }
  }
}
AppEventSource getEventSourceService() => HtmlEventSourceService();