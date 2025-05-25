// lib/utils/rest_api_utility.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data'; // For Uint8List
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kDebugMode;

import 'package:auto_gpt_flutter_client/services/event_source_service.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/api_type.dart';

class RestApiUtility {
  String _serverBaseUrl; // THIS STORES THE ROOT URL, e.g., http://127.0.0.1:8000
  final String _agentApiBasePath = "/ap/v1";
  final String _benchmarkServerBaseUrl = "http://127.0.0.1:8080"; // Assuming different root for benchmark
  final String _benchmarkApiBasePath = "/ap/v1";
  final String _leaderboardBaseUrl = "https://leaderboard.agpt.co"; // Full URL

  RestApiUtility(String serverRootUrl) : _serverBaseUrl = serverRootUrl {
    // Initial normalization of _serverBaseUrl (e.g. remove single trailing slash)
    // The agentBaseUrl getter will perform more robust normalization.
    if (_serverBaseUrl.endsWith('/')) {
      _serverBaseUrl = _serverBaseUrl.substring(0, _serverBaseUrl.length - 1);
    }
    if (kDebugMode) {
      // Note: The agentBaseUrl getter is called here, so it will reflect the new logic.
      print(
          "RestApiUtility initialized with _serverBaseUrl: $_serverBaseUrl. Agent API will be at: $agentBaseUrl");
    }
  }

  void updateBaseURL(String newServerRootUrl) {
    _serverBaseUrl = newServerRootUrl;
    // Initial normalization of _serverBaseUrl
    if (_serverBaseUrl.endsWith('/')) {
      _serverBaseUrl = _serverBaseUrl.substring(0, _serverBaseUrl.length - 1);
    }
    if (kDebugMode) {
      // Note: The agentBaseUrl getter is called here, so it will reflect the new logic.
      print(
          "RestApiUtility _serverBaseUrl updated to: $_serverBaseUrl. Agent API is now at: $agentBaseUrl");
    }
  }

  /// Returns e.g. "http://127.0.0.1:8000/ap/v1" no matter
  /// if _serverBaseUrl was "…:8000" or "…:8000/ap/v1".
  String get agentBaseUrl {
    // Strip all trailing slashes from _serverBaseUrl
    var base = _serverBaseUrl.replaceAll(RegExp(r'/+$'), '');
    // Append _agentApiBasePath only if it’s missing
    if (!base.endsWith(_agentApiBasePath)) {
      base += _agentApiBasePath;
    }
    return base;
  }

  String get benchmarkBaseUrl {
    // This might need similar logic if _benchmarkServerBaseUrl could also include _benchmarkApiBasePath
    return "$_benchmarkServerBaseUrl$_benchmarkApiBasePath";
  }

  String _getEffectiveBaseUrl(ApiType apiType) {
    switch (apiType) {
      case ApiType.agent:
        return agentBaseUrl;
      case ApiType.benchmark:
        return benchmarkBaseUrl;
      case ApiType.leaderboard:
        return _leaderboardBaseUrl;
      default:
        return agentBaseUrl;
    }
  }

  Future<Map<String, dynamic>> get(String endpoint,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final String cleanEndpoint =
        endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
    final url = Uri.parse('$effectiveBaseUrl/$cleanEndpoint');

    if (kDebugMode) {
      print('GET Request to: $url');
    }
    try {
      final response = await http.get(url);
      if (kDebugMode) {
        print('GET Response Status: ${response.statusCode}');
      }
      if (response.statusCode >= 200 && response.statusCode < 300) {
        if (response.body.isEmpty) return {};
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception(
            'Failed to GET from $cleanEndpoint. Status: ${response.statusCode}, Body: ${response.body}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error during GET from $cleanEndpoint: $e');
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>> post(
      String endpoint, Map<String, dynamic> payload,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final String cleanEndpoint =
        endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
    final url = Uri.parse('$effectiveBaseUrl/$cleanEndpoint');

    if (kDebugMode) {
      print('POST Request to: $url');
      print('POST Body: ${jsonEncode(payload)}');
    }
    try {
      final response = await http.post(url,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload));
      if (kDebugMode) {
        print('POST Response Status: ${response.statusCode}');
      }
      if (response.statusCode >= 200 && response.statusCode < 300) {
        if (response.body.isEmpty) return {};
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception(
            'Failed to POST to $cleanEndpoint. Status: ${response.statusCode}, Body: ${response.body}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error during POST to $cleanEndpoint: $e');
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>> put(
      String endpoint, Map<String, dynamic> payload,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final String cleanEndpoint =
        endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
    final url = Uri.parse('$effectiveBaseUrl/$cleanEndpoint');

    if (kDebugMode) {
      print('PUT Request to: $url');
      print('PUT Body: ${jsonEncode(payload)}');
    }
    try {
      final response = await http.put(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      if (kDebugMode) {
        print('PUT Response Status: ${response.statusCode}');
      }
      if (response.statusCode >= 200 && response.statusCode < 300) {
        if (response.body.isEmpty) {
          return {};
        }
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception(
            'Failed to PUT to $cleanEndpoint. Status: ${response.statusCode}, Body: ${response.body}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error during PUT to $cleanEndpoint: $e');
      }
      rethrow;
    }
  }

  Future<Uint8List> getBinary(String endpoint,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final String cleanEndpoint =
        endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
    final url = Uri.parse('$effectiveBaseUrl/$cleanEndpoint');

    if (kDebugMode) {
      print('GET Binary Request to: $url');
    }
    try {
      final response = await http.get(url);
      if (kDebugMode) {
        print('GET Binary Response Status: ${response.statusCode}');
      }
      if (response.statusCode == 200) {
        return response.bodyBytes;
      } else {
        throw Exception(
            'Failed to GET binary from $cleanEndpoint. Status: ${response.statusCode}, Body: ${response.body}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error during GET Binary from $cleanEndpoint: $e');
      }
      rethrow;
    }
  }

  Stream<Map<String, dynamic>> streamEvents(String sseFullUrl) {
    final controller = StreamController<Map<String, dynamic>>();
    AppEventSource eventSource = getEventSourceService();

    if (kDebugMode) {
      print(
          "RestApiUtility: Attempting to connect SSE to: $sseFullUrl (using AppEventSource)");
    }

    eventSource.connect(
      sseFullUrl,
      onOpen: () {
        if (kDebugMode) {
          print("RestApiUtility: SSE stream opened ($sseFullUrl)");
        }
        if (!controller.isClosed) {
          controller.add({'event': 'open', 'data': 'Connection established'});
        }
      },
      onMessage: (String rawData) {
        if (!controller.isClosed) {
          controller.add({'event': 'token_chunk', 'data': rawData});
        }
      },
      onCustomEvent: (String eventName, String rawData) {
        if (!controller.isClosed) {
          try {
            final decodedData = jsonDecode(rawData);
            if (decodedData is Map<String, dynamic>) {
              controller.add({'event': eventName, ...decodedData});
            } else {
              controller.add({'event': eventName, 'data': decodedData});
            }
          } catch (e) {
            if (kDebugMode) {
              print(
                  "RestApiUtility: Error decoding JSON for event '$eventName': $e. Raw data: $rawData");
            }
            controller.add({
              'event': eventName,
              'data': rawData,
              'parsing_error': e.toString()
            });
          }
        }
      },
      onError: () {
        if (kDebugMode) {
          print("RestApiUtility: SSE stream error for $sseFullUrl");
        }
        if (!controller.isClosed) {
          controller.addError(Exception("SSE stream error or closed unexpectedly."));
        }
        // eventSource.close(); // Consider closing based on error type or let onCancel handle
      },
    );

    controller.onCancel = () {
      if (kDebugMode) {
        print(
            "RestApiUtility: SSE stream cancelled by consumer. Closing EventSource for $sseFullUrl.");
      }
      eventSource.close();
      if (!controller.isClosed) {
        controller.close();
      }
    };

    return controller.stream;
  }
}