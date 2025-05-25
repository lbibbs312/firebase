// lib/models/chat.dart
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/models/artifact.dart';

class Chat {
  final String id; // Unique ID for the chat message/event
  final String taskId;
  final String message; // Main textual content
  final DateTime timestamp;
  final MessageType messageType;
  final List<Artifact> artifacts;
  final bool isStreaming; // For agent messages that stream in
  final Map<String, dynamic>? jsonResponse; // Raw JSON from agent if available

  // For MessageType.agentProcess
  final List<String>? processSteps;
  final Map<String, dynamic>? currentThoughtPlanData;
  final bool isProcessComplete;
  final List<String>? webSearchSteps; // <-- NEW: For web search progress

  Chat({
    required this.id,
    required this.taskId,
    required this.message,
    required this.timestamp,
    required this.messageType,
    this.artifacts = const [],
    this.isStreaming = false,
    this.jsonResponse,
    this.processSteps,
    this.currentThoughtPlanData,
    this.isProcessComplete = false,
    this.webSearchSteps, // <-- NEW
  });

  Chat copyWith({
    String? id,
    String? taskId,
    String? message,
    DateTime? timestamp,
    MessageType? messageType,
    List<Artifact>? artifacts,
    bool? isStreaming,
    Map<String, dynamic>? jsonResponse,
    List<String>? processSteps,
    Map<String, dynamic>? currentThoughtPlanData,
    bool? isProcessComplete,
    List<String>? webSearchSteps, // <-- NEW
  }) {
    return Chat(
      id: id ?? this.id,
      taskId: taskId ?? this.taskId,
      message: message ?? this.message,
      timestamp: timestamp ?? this.timestamp,
      messageType: messageType ?? this.messageType,
      artifacts: artifacts ?? this.artifacts,
      isStreaming: isStreaming ?? this.isStreaming,
      jsonResponse: jsonResponse ?? this.jsonResponse,
      processSteps: processSteps ?? this.processSteps,
      currentThoughtPlanData: currentThoughtPlanData ?? this.currentThoughtPlanData,
      isProcessComplete: isProcessComplete ?? this.isProcessComplete,
      webSearchSteps: webSearchSteps ?? this.webSearchSteps, // <-- NEW
    );
  }
}