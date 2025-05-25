// lib/viewmodels/chat_viewmodel.dart
import 'dart:async';
import 'dart:convert';
import 'package:auto_gpt_flutter_client/models/artifact.dart';
import 'package:auto_gpt_flutter_client/models/task_request_body.dart';
import 'package:auto_gpt_flutter_client/services/task_service.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:auto_gpt_flutter_client/services/chat_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'package:auto_gpt_flutter_client/viewmodels/web_search_result.dart';
import 'package:auto_gpt_flutter_client/views/chat/web_search_dialog.dart';


class ChatEvent {
  final String type;
  final String text;
  final Map<String, dynamic>? rawData;
  final List<Artifact>? artifacts;
  final String? streamId;
  final bool isStreamComplete;

  ChatEvent._(this.type, this.text,
      {this.rawData,
      this.artifacts,
      this.streamId,
      this.isStreamComplete = false});

  factory ChatEvent.userMessage(String t) => ChatEvent._('user_message', t);
  factory ChatEvent.loading(String t, {Map<String, dynamic>? raw}) => ChatEvent._('loading', t, rawData: raw);
  factory ChatEvent.finalResponse(String t, {Map<String, dynamic>? raw, List<Artifact>? artifacts}) => ChatEvent._('final_response', t, rawData: raw, artifacts: artifacts);
  factory ChatEvent.error(String t, {Map<String, dynamic>? raw}) => ChatEvent._('error', t, rawData: raw);
  factory ChatEvent.systemMessage(String t, {Map<String, dynamic>? raw}) => ChatEvent._('system_message', t, rawData: raw);
  factory ChatEvent.agentThought(String content, {Map<String, dynamic>? raw}) => ChatEvent._('agent_thought', content, rawData: raw);
  factory ChatEvent.agentPlan(String content, {Map<String, dynamic>? raw, bool isList = false}) {
    if (isList && raw != null && raw['content'] is List) {
      try { List<String> planSteps = List<String>.from(raw['content']); String formattedPlan = planSteps.map((step) => "- $step").join("\n"); return ChatEvent._('agent_plan', formattedPlan, rawData: raw); }
      catch (e) { return ChatEvent._('agent_plan', content, rawData: raw); }
    }
    return ChatEvent._('agent_plan', content, rawData: raw);
  }
  factory ChatEvent.agentCriticism(String content, {Map<String, dynamic>? raw}) => ChatEvent._('agent_criticism', content, rawData: raw);
  factory ChatEvent.agentTool(String toolName, Map<String, dynamic>? arguments, {Map<String, dynamic>? raw}) {
    String argsString = "No arguments";
    if (arguments != null && arguments.isNotEmpty) { try { argsString = JsonEncoder.withIndent('  ').convert(arguments); } catch (e) { argsString = arguments.toString(); }}
    String text = "Tool: $toolName\nArguments:\n$argsString";
    Map<String, dynamic> effectiveRawData = {...?raw, 'tool_name': toolName, 'arguments': arguments};
    return ChatEvent._('agent_tool', text, rawData: effectiveRawData);
  }
  factory ChatEvent.webSearchProgress(String statusType, String detail, {Map<String, dynamic>? raw}) => ChatEvent._('web_search_progress', detail, rawData: {'status_type': statusType, ...?raw});
  factory ChatEvent.actionProposal(String t, {Map<String, dynamic>? raw}) => ChatEvent._('action_proposal', t, rawData: raw);
  factory ChatEvent.streamStart(String id, String initialText, {Map<String, dynamic>? raw}) => ChatEvent._('stream_start', initialText, streamId: id, rawData: raw);
  factory ChatEvent.streamChunk(String id, String chunkText, {Map<String, dynamic>? raw}) => ChatEvent._('stream_chunk', chunkText, streamId: id, rawData: raw);
  factory ChatEvent.streamComplete(String id, String finalText, {Map<String, dynamic>? raw, List<Artifact>? artifacts}) => ChatEvent._('stream_complete', finalText, streamId: id, rawData: raw, artifacts: artifacts, isStreamComplete: true);
  factory ChatEvent.agentAwaitingInput(String question, String stepIdAwaitingReply, {Map<String, dynamic>? raw}) => ChatEvent._('agent_awaiting_input', question, rawData: {'question': question, 'step_id_awaiting_reply': stepIdAwaitingReply, ...?raw});

  @override
  String toString() {
    String shortText = text.length > 60 ? "${text.substring(0, 60)}..." : text;
    bool hasArtifactsList = artifacts != null && artifacts!.isNotEmpty;
    return "ChatEvent(type: $type, text: '$shortText', streamId: $streamId, streamComplete: $isStreamComplete, hasRawData: ${rawData != null}, hasArtifacts: $hasArtifactsList)";
  }
}

class ChatViewModel with ChangeNotifier {
  final ChatService _chatService;
  final TaskService _taskService;
  final SharedPreferencesService _prefsService;

  String? _currentTaskId;
  bool _isContinuousMode = false;
  bool _isWaitingForAgentResponse = false;

  bool _isAgentCurrentlyAwaitingUserInput = false;
  String? _currentAgentQuestionForUser;
  String? _stepIdAwaitingReplyFromUser;

  StreamSubscription? _sseSubscription;
  final StreamController<ChatEvent> _eventController = StreamController<ChatEvent>.broadcast();
  Stream<ChatEvent> get events => _eventController.stream;

  String? _activeAgentStreamId;

  ChatViewModel(this._chatService, this._taskService, this._prefsService) {
    if (kDebugMode) {
      print(">> ChatViewModel CREATED (Consumes stream from ChatService)");
    }
  }

  String? get currentTaskId => _currentTaskId;
  bool get isWaitingForAgentResponse => _isWaitingForAgentResponse;
  SharedPreferencesService get prefsService => _prefsService;
  bool get isAgentCurrentlyAwaitingUserInput => _isAgentCurrentlyAwaitingUserInput;
  String? get currentAgentQuestionForUser => _currentAgentQuestionForUser;

  bool get isContinuousMode => _isContinuousMode;
  set isContinuousMode(bool value) {
    if (_isContinuousMode != value) {
      _isContinuousMode = value;
      if (kDebugMode) {
        print(">> ChatViewModel.isContinuousMode set to: $value");
      }
      notifyListeners();
    }
  }

  void _initiateAgentSpeechStream(String streamId, String initialText, Map<String, dynamic>? rawData) {
    _activeAgentStreamId = streamId;
    _eventController.add(ChatEvent.streamStart(streamId, initialText, raw: rawData));
  }

  void _appendAgentSpeechChunk(String streamId, String chunk, Map<String, dynamic>? rawData) {
    if (_activeAgentStreamId == streamId || _activeAgentStreamId == null) {
      if (_activeAgentStreamId == null && chunk.isNotEmpty) {
        _initiateAgentSpeechStream(streamId, chunk, rawData);
      } else if (_activeAgentStreamId != null) {
        _eventController.add(ChatEvent.streamChunk(streamId, chunk, raw: rawData));
      }
    } else {
      if (kDebugMode) {
        print(">> ChatViewModel: Received chunk for inactive/different stream ID: $streamId (active: $_activeAgentStreamId)");
      }
    }
  }

  void _completeAgentSpeechStream(String streamId, String finalText, Map<String, dynamic>? rawData, List<Artifact>? artifacts) {
    _eventController.add(ChatEvent.streamComplete(streamId, finalText, raw: rawData, artifacts: artifacts));
    if (_activeAgentStreamId == streamId) {
      _activeAgentStreamId = null;
    }
  }

  Future<void> _setActiveTaskAndFetchHistoryInternal(String taskId) async {
    if (_currentTaskId != taskId) {
      _currentTaskId = taskId;
      await _closeSseSubscription();
      _isWaitingForAgentResponse = true;
      _isAgentCurrentlyAwaitingUserInput = false;
      notifyListeners();
      _eventController.add(ChatEvent.systemMessage("Loading history for task $taskId..."));
      await _fetchHistoricalMessagesAsEventsInternal();
      _isWaitingForAgentResponse = false;
      notifyListeners();
    }
  }

  void _clearActiveTaskInternal() {
    _currentTaskId = null;
    _closeSseSubscription();
    _isWaitingForAgentResponse = false;
    _activeAgentStreamId = null;
    _isAgentCurrentlyAwaitingUserInput = false;
    _eventController.add(ChatEvent.systemMessage("Task cleared."));
    notifyListeners();
  }

  Future<void> _fetchHistoricalMessagesAsEventsInternal() async {
    if (_currentTaskId == null) return;
    _eventController.add(ChatEvent.loading("Fetching history for task '$_currentTaskId'..."));
    try {
      final Map<String, dynamic> stepsResponse = await _chatService.listTaskSteps(_currentTaskId!, pageSize: 10000);
      final List<dynamic> stepsJsonList = stepsResponse['steps'] ?? [];
      for (Map<String, dynamic> stepMap_dynamic in stepsJsonList) {
        final Map<String, dynamic> stepMap = Map<String, dynamic>.from(stepMap_dynamic);
        List<Artifact> stepArtifacts = [];
        if (stepMap['artifacts'] != null && stepMap['artifacts'] is List) {
          stepArtifacts = (stepMap['artifacts'] as List).map((art) => Artifact.fromJson(Map<String, dynamic>.from(art as Map))).toList();
        }
        if (stepMap['input']?.toString().isNotEmpty ?? false) {
          _eventController.add(ChatEvent.userMessage(stepMap['input'].toString()));
        }
        if (stepMap['status'] == 'awaiting_user_input' && (stepMap['output']?.toString().isNotEmpty ?? false)) {
          String question = stepMap['output'].toString().replaceFirst("Agent asked: ", "");
          _eventController.add(ChatEvent.agentAwaitingInput(question, stepMap['step_id'].toString(), raw: stepMap));
        } else if (stepMap['output']?.toString().isNotEmpty ?? false) {
          _eventController.add(ChatEvent.finalResponse(stepMap['output'].toString(), raw: stepMap, artifacts: stepArtifacts));
        }
      }
      _eventController.add(ChatEvent.systemMessage("History loaded."));
    } catch (error) {
      _eventController.add(ChatEvent.error("Error loading history: $error"));
    }
  }

  Future<void> setCurrentTaskId(String taskId) async {
    await _setActiveTaskAndFetchHistoryInternal(taskId);
  }

  Future<void> fetchChatsForTask() async {
    await _fetchHistoricalMessagesAsEventsInternal();
  }

  void clearCurrentTaskAndChats() {
    _clearActiveTaskInternal();
  }

  Future<void> sendReplyToAgent(String userAnswer) async {
    if (!_isAgentCurrentlyAwaitingUserInput || _currentTaskId == null || _stepIdAwaitingReplyFromUser == null) {
      _eventController.add(ChatEvent.error("Cannot send reply: No agent question pending or task ID missing."));
      return;
    }
    _eventController.add(ChatEvent.userMessage(userAnswer.trim()));
    _isAgentCurrentlyAwaitingUserInput = false;
    _currentAgentQuestionForUser = null;
    _stepIdAwaitingReplyFromUser = null;
    _isWaitingForAgentResponse = true;
    _activeAgentStreamId = null;
    notifyListeners();
    await _initiateAgentStepSSE(taskId: _currentTaskId!, agentInput: userAnswer.trim());
  }

  Future<void> sendUserMessage(String userMessage, {int continuousModeSteps = 1, int currentStep = 1}) async {
    if (_isAgentCurrentlyAwaitingUserInput) {
      await sendReplyToAgent(userMessage);
      return;
    }
    _eventController.add(ChatEvent.userMessage(userMessage.trim()));
    _isWaitingForAgentResponse = true;
    _activeAgentStreamId = null;
    notifyListeners();
    String effectiveTaskId = _currentTaskId ?? "";
    final String sseAgentInput = userMessage.trim();
    try {
      if (_currentTaskId == null || _currentTaskId!.isEmpty) {
        _eventController.add(ChatEvent.loading("Creating task..."));
        final taskRequestBody = TaskRequestBody(input: sseAgentInput.isNotEmpty ? sseAgentInput : "New Task Session");
        final Map<String, dynamic> taskCreationResponse = await _taskService.createTask(taskRequestBody);
        effectiveTaskId = taskCreationResponse['task_id']?.toString() ?? "";
        if (effectiveTaskId.isEmpty) throw Exception("Task creation failed: 'task_id' missing.");
        _currentTaskId = effectiveTaskId;
        if (kDebugMode) {
          print(">> Task '$_currentTaskId' created. (ViewModel log)");
        }
      } else {
        effectiveTaskId = _currentTaskId!;
      }
      String inputForStep = (sseAgentInput.isEmpty && currentStep > 1 && _isContinuousMode) ? "" : sseAgentInput;
      await _initiateAgentStepSSE(taskId: effectiveTaskId, agentInput: inputForStep, continuousModeSteps: continuousModeSteps, currentStep: currentStep);
    } catch (e, stackTrace) {
      _handleSSEError('Failed to send message: ${e.toString()}', stackTrace);
    }
  }

  void handleAgentToolResult(String toolName, Map<String, dynamic>? toolArguments, Map<String, dynamic> toolResultData, BuildContext context) {
    if (kDebugMode) {
      print(">> ChatViewModel: Handling tool result for '$toolName'");
    }
    if (toolName == 'web_search' || toolName == 'search') {
      try {
        final searchResult = WebSearchResult.fromJson(toolResultData);
        showWebSearchOverlay(context, searchResult);
      } catch (e, s) {
        if (kDebugMode) {
          print("Error parsing web search result or showing overlay: $e\n$s");
        }
        _eventController.add(ChatEvent.error("Failed to display web search results: $e"));
      }
    } else {
      if (kDebugMode) {
        print(">> ChatViewModel: Unhandled tool result for '$toolName'");
      }
      String resultSummary = toolResultData.toString();
      if (resultSummary.length > 200) resultSummary = resultSummary.substring(0, 200) + "...";
      _eventController.add(ChatEvent.systemMessage("Tool '$toolName' executed. Result: $resultSummary"));
    }
  }

  Future<void> _initiateAgentStepSSE(
      {required String taskId,
      required String agentInput,
      Map<String, dynamic>? additionalInputForBackend,
      int continuousModeSteps = 1,
      int currentStep = 1}) async {
    
    await _closeSseSubscription();

    _eventController.add(ChatEvent.loading('Connecting to agent stream...'));

    _sseSubscription = _chatService.streamStepExecution(taskId, agentInput.isNotEmpty ? agentInput : null)
      .listen(
        (Map<String, dynamic> eventMap) {
          final String eventType = eventMap['event'] as String? ?? 'unknown';
          final dynamic eventDataPayload = eventMap['data'];
          final Map<String, dynamic> data = (eventDataPayload is Map)
              ? Map<String, dynamic>.from(eventDataPayload)
              : (eventMap..remove('event'));

          if (kDebugMode && eventType != 'token_chunk' && eventType != 'progress') {
             print(">> ChatViewModel Received SSE Event: type='$eventType', payloadKeys='${data.keys.toList()}'");
          }
          
          if (_isWaitingForAgentResponse && eventType != 'progress' && eventType != 'token_chunk') {
            _isWaitingForAgentResponse = false;
            notifyListeners();
          }

          if (eventType == 'token_chunk' && eventDataPayload is String) {
              String streamId = _activeAgentStreamId ?? "default_final_stream_${DateTime.now().millisecondsSinceEpoch}";
              _appendAgentSpeechChunk(streamId, eventDataPayload, data);
          } else if (eventType == 'ask_user') {
            final question = data['question'] as String? ?? "Clarification needed";
            final stepIdAwaitingReply = data['step_id_awaiting_reply'] as String? ?? "unknown_step";
            _eventController.add(ChatEvent.agentAwaitingInput(question, stepIdAwaitingReply, raw: data));
            _isAgentCurrentlyAwaitingUserInput = true; _currentAgentQuestionForUser = question; _stepIdAwaitingReplyFromUser = stepIdAwaitingReply;
            _activeAgentStreamId = null; notifyListeners();
            if (kDebugMode) { print(">> SSE 'ask_user' received. Question: '$question'.");}
          } else if (eventType == 'progress') {
            _eventController.add(ChatEvent.loading(data['message']?.toString() ?? 'Processing...', raw: data));
          } else if (eventType == 'agent_speech') {
            String speechText = data['message']?.toString() ?? eventDataPayload?.toString() ?? "Agent speech";
            _eventController.add(ChatEvent.finalResponse(speechText, raw: data));
          } else if (eventType == 'agent_thought_event') {
            String content = data['content']?.toString() ?? data['thought']?.toString() ?? eventDataPayload?.toString() ?? "Agent thought";
            _eventController.add(ChatEvent.agentThought(content, raw: data));
          } else if (eventType == 'agent_plan_event') {
            dynamic content = data['content'] ?? data['plan'];
            if (content is List) { _eventController.add(ChatEvent.agentPlan("", raw: {'content': content, ...data}, isList: true)); }
            else { _eventController.add(ChatEvent.agentPlan(content?.toString() ?? eventDataPayload?.toString() ?? "Agent plan", raw: data)); }
          } else if (eventType == 'agent_criticism_event') {
            String content = data['content']?.toString() ?? data['criticism']?.toString() ?? eventDataPayload?.toString() ?? "Agent criticism";
            _eventController.add(ChatEvent.agentCriticism(content, raw: data));
          } else if (eventType == 'agent_tool_event') {
            String toolName = data['tool_name'] as String? ?? data['name'] as String? ?? "Unknown Tool";
            Map<String, dynamic>? arguments = data['arguments'] is Map ? Map<String, dynamic>.from(data['arguments']) : null;
            _eventController.add(ChatEvent.agentTool(toolName, arguments, raw: data));
          } else if (eventType == 'web_search_progress_event') {
            String statusType = data['type'] as String? ?? data['status_type'] as String? ?? 'unknown_web_search_status';
            String detail = "";
            if (statusType == 'web_search_query') { detail = "Searching for: \"${data['query']}\""; }
            else if (statusType == 'web_search_fetching') { detail = "${data['status'] ?? 'Fetching'}: ${data['url']}"; }
            else { detail = data['message'] as String? ?? eventDataPayload?.toString() ?? "Web search update"; }
            _eventController.add(ChatEvent.webSearchProgress(statusType, detail, raw: data));
          } else if (eventType == 'sse_stream_step_completed') {
              final Map<String, dynamic> stepDetails = data['final_db_step_details'] is Map 
                  ? Map<String, dynamic>.from(data['final_db_step_details']) 
                  : data;

              if (stepDetails.containsKey('tool_executed') && stepDetails['tool_executed'] is Map) {
                  Map<String, dynamic> toolExecutionInfo = Map<String, dynamic>.from(stepDetails['tool_executed']);
                  String executedToolName = toolExecutionInfo['name'] as String? ?? "unknown";
                  Map<String, dynamic>? toolArguments = toolExecutionInfo['arguments'] is Map ? Map<String, dynamic>.from(toolExecutionInfo['arguments']) : null;
                  Map<String, dynamic> toolResultPayload = toolExecutionInfo['result'] is Map
                      ? Map<String, dynamic>.from(toolExecutionInfo['result'])
                      : {'output': toolExecutionInfo['result']?.toString() ?? "No result data."};
                  
                  _eventController.add(
                    ChatEvent._(
                      'tool_result_available',
                      "Result for tool '$executedToolName' available.",
                      rawData: {
                        'tool_name': executedToolName,
                        'tool_arguments': toolArguments,
                        'tool_result_data': toolResultPayload,
                      }
                    )
                  );
              }

              String finalStepOutput = stepDetails['output']?.toString() ?? 'Step completed.';
              List<Artifact> arts = [];
              if (stepDetails['artifacts'] is List) { arts = (stepDetails['artifacts'] as List).map((art) => Artifact.fromJson(Map<String, dynamic>.from(art as Map))).toList(); }
              String stepId = stepDetails['step_id']?.toString() ?? _activeAgentStreamId ?? "completed_step_${DateTime.now().millisecondsSinceEpoch}";

              if (_activeAgentStreamId != null && _activeAgentStreamId == stepId) {
                _completeAgentSpeechStream(stepId, finalStepOutput, stepDetails, arts);
              } else if (!finalStepOutput.contains("Interaction cycle ended.") && !finalStepOutput.contains("Proposal was processed.")) {
                   if (kDebugMode) {print(">> SSE 'sse_stream_step_completed': Output: $finalStepOutput.");}
              }
              _isWaitingForAgentResponse = false;
              _activeAgentStreamId = null;
              notifyListeners();
              _closeSseSubscription();
              bool isLastByAgent = stepDetails['is_last'] as bool? ?? false;
              if (_isContinuousMode && !isLastByAgent && currentStep < continuousModeSteps) {
                sendUserMessage("", continuousModeSteps: continuousModeSteps, currentStep: currentStep + 1);
              } else if (_isContinuousMode) {
                isContinuousMode = false;
                _eventController.add(ChatEvent.systemMessage(isLastByAgent ? "Continuous mode finished: Agent completed task." : "Continuous mode finished. Max steps reached or task ended."));
              }
          } else if (eventType == 'open') {
            if (kDebugMode) {print("ChatViewModel: SSE Stream Opened (event from stream)");}
          } else {
            if (kDebugMode) {print(">> ChatViewModel: Received unhandled SSE event type '$eventType' from stream: $data");}
          }
        },
        onError: (error, stackTrace) {
          if (kDebugMode) {
            print("ChatViewModel: SSE stream error: $error");
          }
          _handleSSEError('Stream connection error.', stackTrace, rawData: {'error_object': error.toString()});
          _sseSubscription = null;
        },
        onDone: () {
          if (kDebugMode) {
            print("ChatViewModel: SSE stream done/closed by source.");
          }
          if (_isWaitingForAgentResponse || _activeAgentStreamId != null) {
              if (!_isAgentCurrentlyAwaitingUserInput) {
                   _handleSSEError('Stream closed by server unexpectedly (onDone).', null);
              }
          }
          _isWaitingForAgentResponse = false;
          _activeAgentStreamId = null;
          notifyListeners();
          _sseSubscription = null;
        },
      );
  }

  Future<void> _closeSseSubscription() async {
    if (_sseSubscription != null) {
      if (kDebugMode) {
        print(">> ChatViewModel: Cancelling existing SSE subscription.");
      }
      await _sseSubscription!.cancel();
      _sseSubscription = null;
    }
  }

  void _handleSSEError(String errorMessage, StackTrace? stackTrace, {Map<String, dynamic>? rawData}) {
    _eventController.add(ChatEvent.error(errorMessage, raw: rawData));
    _isWaitingForAgentResponse = false; _activeAgentStreamId = null; _isAgentCurrentlyAwaitingUserInput = false;
    if (_isContinuousMode) { isContinuousMode = false; _eventController.add(ChatEvent.systemMessage("Continuous mode stopped due to an error.")); }
    notifyListeners(); 
    _closeSseSubscription();
    if (kDebugMode) { print(">> ChatViewModel._handleSSEError: $errorMessage"); if (stackTrace != null) { print(stackTrace); }}
  }

  Future<void> downloadArtifact(String taskId, String artifactId) async {
    _eventController.add(ChatEvent.loading("Initiating download for $artifactId..."));
    try { 
      await _chatService.downloadArtifact(taskId, artifactId); 
      _eventController.add(ChatEvent.systemMessage("Download for $artifactId initiated (browser/OS will handle)."));
    } catch (e) { 
      _eventController.add(ChatEvent.error("Failed to download $artifactId: $e")); 
    }
  }

  @override
  void dispose() {
    _closeSseSubscription();
    _eventController.close();
    super.dispose();
  }
}