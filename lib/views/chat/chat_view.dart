// lib/views/chat/chat_view.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fluttertoast/fluttertoast.dart';

import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';

import 'package:auto_gpt_flutter_client/views/chat/agent_message_tile.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_input_field.dart';
import 'package:auto_gpt_flutter_client/views/chat/loading_indicator.dart';
import 'package:auto_gpt_flutter_client/views/chat/user_message_tile.dart';

class ChatView extends StatefulWidget {
  const ChatView({Key? key}) : super(key: key);

  @override
  _ChatViewState createState() => _ChatViewState();
}

class _ChatViewState extends State<ChatView> {
  final ScrollController _scrollController = ScrollController();
  late ChatViewModel _chatViewModel;
  StreamSubscription? _eventSubscription;
  final List<Chat> _displayedMessages = [];

  String _currentGlobalStatusText = "Thinking...";
  String? _currentGlobalThoughtText;
  String? _activeAgentProcessMessageId;

  MessageType _parseChatEventTypeToMessageType(String eventTypeString) {
    switch (eventTypeString) {
      case 'user_message':
        return MessageType.userMessage;
      case 'loading':
        return MessageType.loading;
      case 'agent_thought':
        return MessageType.agentThought;
      case 'agent_plan':
        return MessageType.agentPlan;
      case 'agent_criticism':
        return MessageType.agentCriticism;
      case 'agent_tool':
        return MessageType.agentTool;
      case 'action_proposal':
        return MessageType.actionProposal;
      case 'final_response':
        return MessageType.finalResponse;
      case 'agent_speech': // Treat as final response for standalone speech
        return MessageType.finalResponse;
      case 'error':
        return MessageType.error;
      case 'system_message':
        return MessageType.systemMessage;
      case 'agent_awaiting_input':
        return MessageType.agent; // For direct agent questions
      case 'stream_start':
      case 'stream_chunk':
      case 'stream_complete':
        return MessageType.finalResponse; // For final response streaming
      default:
        print(
            ">> ChatView: Unknown ChatEvent type string '$eventTypeString', defaulting to MessageType.systemInfo.");
        return MessageType.systemInfo;
    }
  }

  void _finalizeActiveAgentProcessBubble({String? newMessageHeader}) {
    if (_activeAgentProcessMessageId != null) {
      int processIdx = _displayedMessages
          .indexWhere((m) => m.id == _activeAgentProcessMessageId);
      if (processIdx != -1) {
        _displayedMessages[processIdx] =
            _displayedMessages[processIdx].copyWith(
          isProcessComplete: true,
          message: newMessageHeader ?? _displayedMessages[processIdx].message,
        );
      }
      _activeAgentProcessMessageId = null;
    }
  }

  void _upsertAgentProcessBubble(ChatEvent event) {
    String initialHeader = "Agent is processing...";
    Map<String, dynamic>? initialData;
    List<String> initialSteps = [];

    if (event.type == 'loading') {
      initialHeader = event.text;
      initialSteps.add(event.text);
    } else if (event.rawData != null) {
      initialData = Map<String, dynamic>.from(event.rawData!);
      if (event.type == 'action_proposal' &&
          event.rawData?['thoughts']?['speak'] != null) {
        initialHeader = event.rawData!['thoughts']['speak'];
      } else if (event.type == 'agent_thought') {
        initialHeader = "Agent is thinking...";
      }
    }

    if (_activeAgentProcessMessageId == null) {
      _activeAgentProcessMessageId =
          "agent_process_${DateTime.now().millisecondsSinceEpoch}";
      _displayedMessages.add(Chat(
        id: _activeAgentProcessMessageId!,
        taskId: _chatViewModel.currentTaskId ?? "process_task",
        message: initialHeader,
        timestamp: DateTime.now(),
        messageType: MessageType.agentProcess,
        processSteps: initialSteps,
        currentThoughtPlanData: initialData,
        jsonResponse: event.rawData,
        isProcessComplete: false,
      ));
      _currentGlobalStatusText = initialHeader;
      _currentGlobalThoughtText =
          (event.type == 'agent_thought') ? event.text : null;
    } else {
      int idx = _displayedMessages
          .indexWhere((m) => m.id == _activeAgentProcessMessageId);
      if (idx != -1) {
        Chat existingChat = _displayedMessages[idx];
        List<String> updatedSteps =
            List.from(existingChat.processSteps ?? []);
        Map<String, dynamic> updatedThoughtPlanData =
            Map<String, dynamic>.from(existingChat.currentThoughtPlanData ?? {});
        String updatedHeader = existingChat.message;

        if (event.type == 'loading' && event.text.isNotEmpty) {
          if (updatedSteps.isEmpty || updatedSteps.last != event.text) {
            updatedSteps.add(event.text);
          }
          updatedHeader = event.text;
        } else if (event.type == 'agent_thought') {
          updatedThoughtPlanData['thoughts'] = {
            'text': event.text,
            ...?(event.rawData?['thoughts'] as Map?)
          };
          updatedHeader = "Agent is thinking...";
        } else if (event.type == 'agent_plan') {
          updatedThoughtPlanData['plan'] =
              event.rawData?['content'] ?? (event.text.split('\n'));
        } else if (event.type == 'agent_criticism') {
          Map<String, dynamic> thoughtsMap =
              Map<String, dynamic>.from(updatedThoughtPlanData['thoughts'] ?? {});
          thoughtsMap['self_criticism'] = event.text;
          updatedThoughtPlanData['thoughts'] = thoughtsMap;
        } else if (event.type == 'agent_tool') {
          // This case is for adding tool info to the *process bubble itself*
          updatedThoughtPlanData['use_tool'] = event.rawData ??
              {'name': event.text.split(':').last.split('\n').first.trim()};
          if (event.rawData?['tool_name'] != null) {
            updatedHeader = "Using tool: ${event.rawData!['tool_name']}";
          }
        } else if (event.type == 'action_proposal' && event.rawData != null) {
          (event.rawData as Map<String, dynamic>).forEach((key, value) {
            if (updatedThoughtPlanData.containsKey(key) &&
                updatedThoughtPlanData[key] is Map &&
                value is Map) {
              (updatedThoughtPlanData[key] as Map).addAll(value as Map);
            } else {
              updatedThoughtPlanData[key] = value;
            }
          });
          if (event.rawData?['thoughts']?['speak'] != null) {
            updatedHeader = event.rawData!['thoughts']['speak'];
          }
        }

        _displayedMessages[idx] = existingChat.copyWith(
          message: updatedHeader,
          processSteps: updatedSteps,
          currentThoughtPlanData: updatedThoughtPlanData,
          timestamp: DateTime.now(),
          jsonResponse: event.rawData ?? existingChat.jsonResponse,
        );
        _currentGlobalStatusText = updatedHeader;
        _currentGlobalThoughtText = (event.type == 'agent_thought')
            ? event.text
            : _currentGlobalThoughtText;
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _chatViewModel = Provider.of<ChatViewModel>(context, listen: false);

    if (_chatViewModel.currentTaskId != null) {
      _chatViewModel.fetchChatsForTask();
    }

    _eventSubscription = _chatViewModel.events.listen((event) async {
      if (!mounted) return;

      // --- ARTIFICIAL DELAYS ---
      bool isNewAgentProcessBubbleNeeded = (event.type == 'agent_thought' ||
              event.type == 'agent_plan' ||
              event.type == 'agent_criticism' ||
              (event.type == 'agent_tool' &&
                  event.rawData?['tool_name'] != 'web_search') || // Don't delay for web search UI tool bubble
              (event.type == 'action_proposal' && event.rawData != null)) &&
          _activeAgentProcessMessageId == null;

      if (isNewAgentProcessBubbleNeeded) {
        await Future.delayed(const Duration(milliseconds: 400));
      } else if (event.type == 'final_response' ||
          event.type == 'agent_speech' ||
          (event.type == 'stream_complete' && event.text.isNotEmpty)) {
        if (_activeAgentProcessMessageId != null) {
          await Future.delayed(const Duration(milliseconds: 1200));
        } else {
          await Future.delayed(const Duration(milliseconds: 300));
        }
      }
      if (!mounted) return;
      // --- END ARTIFICIAL DELAYS ---

      setState(() {
        String eventSpecificId = event.rawData?['step_id']?.toString() ??
            event.streamId ??
            "${event.type}_${DateTime.now().millisecondsSinceEpoch}";

        MessageType parsedMsgType =
            _parseChatEventTypeToMessageType(event.type);

        // Global Indicator Updates (pre-message handling)
        if (_chatViewModel.isWaitingForAgentResponse &&
            _activeAgentProcessMessageId == null) {
          if (event.type == 'loading') {
            _currentGlobalStatusText = event.text;
            _currentGlobalThoughtText = null;
          } else if (event.type == 'agent_thought' || event.type == 'thought') {
            _currentGlobalThoughtText = event.text;
            _currentGlobalStatusText = "Agent is thinking...";
          }
        }

        if (event.type == 'user_message') {
          _finalizeActiveAgentProcessBubble();
          _currentGlobalStatusText = "Thinking...";
          _currentGlobalThoughtText = null;
          _displayedMessages.add(Chat(
              id: eventSpecificId,
              taskId: _chatViewModel.currentTaskId ?? "pending_task",
              message: event.text,
              timestamp: DateTime.now(),
              messageType: parsedMsgType));
        } else if (event.type == 'agent_tool' &&
            event.rawData?['tool_name'] == 'web_search' &&
            event.rawData?['arguments']?['query'] != null) {
          // Handle Web Search:
          // 1. Ensure agentProcess bubble reflects tool usage
          _upsertAgentProcessBubble(event); // This will add/update the 'use_tool' in agentProcess

          // 2. Add the separate MessageType.agentTool message for the WebView
          _displayedMessages.add(Chat(
            id:
                "web_search_ui_${event.rawData?['step_id'] ?? DateTime.now().millisecondsSinceEpoch}",
            taskId: _chatViewModel.currentTaskId ?? "tool_task",
            message:
                "Searching for: \"${event.rawData!['arguments']['query']}\"",
            timestamp: DateTime.now(),
            messageType: MessageType.agentTool, // Triggers WebView
            jsonResponse: event.rawData,
          ));
          _currentGlobalStatusText =
              "Searching: \"${event.rawData!['arguments']['query']}\"";
          _currentGlobalThoughtText = null;
        } else if (event.type == 'loading' ||
            event.type == 'agent_thought' ||
            event.type == 'agent_plan' ||
            event.type == 'agent_criticism' ||
            (event.type == 'agent_tool' &&
                event.rawData?['tool_name'] !=
                    'web_search') || // Other tools update process bubble
            (event.type == 'action_proposal' &&
                event.rawData != null &&
                (event.rawData!['thoughts'] != null ||
                    event.rawData!['plan'] != null ||
                    event.rawData!['use_tool'] != null))) {
          // Generic handling for creating/updating the AgentProcess Bubble
          _upsertAgentProcessBubble(event);
        } else if (event.type == 'final_response' ||
            event.type == 'agent_speech' ||
            (event.type == 'stream_complete' && event.text.isNotEmpty) ||
            event.type == 'error' ||
            (event.type == 'action_proposal' &&
                (event.rawData == null ||
                    !(event.rawData!['thoughts'] != null ||
                        event.rawData!['plan'] != null ||
                        event.rawData!['use_tool'] != null)))) {
          // Finalize active process bubble before showing final message
          _finalizeActiveAgentProcessBubble(
              newMessageHeader: (event.type == 'error')
                  ? "Error occurred"
                  : "Processing complete");

          _currentGlobalStatusText = "Done.";
          _currentGlobalThoughtText = null;

          _displayedMessages.add(Chat(
            id: eventSpecificId,
            taskId: _chatViewModel.currentTaskId ?? "final_task",
            message: event.text,
            timestamp: DateTime.now(),
            messageType: (event.type == 'error')
                ? MessageType.error
                : MessageType.finalResponse,
            jsonResponse: event.rawData,
            artifacts: event.artifacts ?? const [],
            isStreaming: false, // Streaming handled by stream_ events
          ));
        } else if (event.type == 'system_message') {
          if (event.text != '✅ Agent Connected. Waiting for response...') {
            _displayedMessages.add(Chat(
                id: eventSpecificId,
                taskId: _chatViewModel.currentTaskId ?? "system_task",
                message: event.text,
                timestamp: DateTime.now(),
                messageType: parsedMsgType));
          }
        } else if (event.type == 'agent_awaiting_input') {
          _finalizeActiveAgentProcessBubble(
              newMessageHeader: "Awaiting your input");
          _displayedMessages.add(Chat(
              id: eventSpecificId,
              taskId: _chatViewModel.currentTaskId ?? "ask_task",
              message: "🤔 ${event.text}",
              timestamp: DateTime.now(),
              messageType: MessageType.agent,
              jsonResponse: event.rawData));
        } else if (event.type.startsWith('stream_')) {
          int msgIdx = _displayedMessages.lastIndexWhere((m) =>
              m.id == event.streamId &&
              (m.messageType == MessageType.agent ||
                  m.messageType == MessageType.finalResponse));
          if (msgIdx != -1) {
            _displayedMessages[msgIdx] =
                _displayedMessages[msgIdx].copyWith(
              message: event.type == 'stream_start'
                  ? event.text
                  : _displayedMessages[msgIdx].message + event.text,
              isStreaming: !event.isStreamComplete,
              messageType: MessageType.finalResponse,
              artifacts: event.isStreamComplete
                  ? (event.artifacts ?? _displayedMessages[msgIdx].artifacts)
                  : _displayedMessages[msgIdx].artifacts,
              timestamp: DateTime.now(),
              jsonResponse: event.isStreamComplete
                  ? (event.rawData ?? _displayedMessages[msgIdx].jsonResponse)
                  : _displayedMessages[msgIdx].jsonResponse,
            );
          } else if (event.type == 'stream_start') {
            // If a stream starts, finalize any existing process bubble.
            _finalizeActiveAgentProcessBubble(
                newMessageHeader: "Generating response...");
            _displayedMessages.add(Chat(
                id: event.streamId!,
                taskId: _chatViewModel.currentTaskId ?? "streaming_task",
                message: event.text,
                timestamp: DateTime.now(),
                messageType: MessageType.finalResponse,
                isStreaming: true,
                jsonResponse: event.rawData));
          }
          if (event.isStreamComplete) {
            _finalizeActiveAgentProcessBubble(
                newMessageHeader: "Response complete");
            _currentGlobalStatusText = "Done.";
            _currentGlobalThoughtText = null;
          }
        } else {
          print(
              ">> ChatView [${DateTime.now()}] Unhandled Event (default case): type='${event.type}', text='${event.text.length > 50 ? event.text.substring(0, 50) + "..." : event.text}'");
          if (parsedMsgType != MessageType.loading) {
            _displayedMessages.add(Chat(
              id: eventSpecificId,
              taskId: _chatViewModel.currentTaskId ?? "unknown_event",
              message: "[${event.type}] ${event.text}",
              timestamp: DateTime.now(),
              messageType: MessageType.systemInfo,
              jsonResponse: event.rawData,
            ));
          }
        }
        _scrollToBottom();
      });
    }, onError: (error, stackTrace) {
      if (mounted) {
        setState(() {
          _finalizeActiveAgentProcessBubble(
              newMessageHeader: "Critical Stream Error!");
          _currentGlobalStatusText = "Critical Stream Error!";
          _currentGlobalThoughtText = null;
          _displayedMessages.add(Chat(
              id:
                  "stream_listen_fatal_err_${DateTime.now().millisecondsSinceEpoch}",
              taskId: _chatViewModel.currentTaskId ?? "unknown_task_fatal",
              message: "Critical error listening to agent: $error",
              timestamp: DateTime.now(),
              messageType: MessageType.error,
              artifacts: const []));
          _scrollToBottom();
        });
      }
    });
    _scrollController.addListener(_scrollListener);
  }

  void _scrollListener() {}

  void _scrollToBottom() {
    if (_scrollController.hasClients && _displayedMessages.isNotEmpty) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted && _scrollController.hasClients) {
          _scrollController.animateTo(
            _scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  void dispose() {
    _eventSubscription?.cancel();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _onSendPressed(String message) async {
    final String trimmedMessage = message.trim();
    if (trimmedMessage.isEmpty) return;
    if (mounted) {
      setState(() {
        _currentGlobalStatusText = "Thinking...";
        _currentGlobalThoughtText = null;
      });
    }
    try {
      await _chatViewModel.sendUserMessage(
        trimmedMessage,
        continuousModeSteps:
            Provider.of<SettingsViewModel>(context, listen: false)
                .continuousModeSteps,
      );
    } catch (e) {
      Fluttertoast.showToast(msg: "Failed to send: $e");
      if (mounted) {
        setState(() {
          if (_chatViewModel.isWaitingForAgentResponse) {
            _currentGlobalStatusText = "Error sending message.";
          }
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final chatViewModelReader = Provider.of<ChatViewModel>(context);

    bool showGlobalThinkingIndicator =
        chatViewModelReader.isWaitingForAgentResponse &&
            !chatViewModelReader.isAgentCurrentlyAwaitingUserInput &&
            _activeAgentProcessMessageId ==
                null; // Only show if no process bubble is active

    bool showAgentConnectedMessage =
        !chatViewModelReader.isWaitingForAgentResponse &&
            !chatViewModelReader.isAgentCurrentlyAwaitingUserInput &&
            _displayedMessages
                .where((m) => m.messageType != MessageType.userMessage)
                .isEmpty; // Show if no agent messages yet

    return Scaffold(
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _displayedMessages.length,
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              itemBuilder: (context, index) {
                final chat = _displayedMessages[index];
                if (chat.messageType == MessageType.userMessage) {
                  return UserMessageTile(message: chat.message);
                } else {
                  return AgentMessageTile(
                    key: ValueKey(
                        "${chat.id}_${chat.timestamp.millisecondsSinceEpoch}_${chat.message.hashCode}_${chat.isStreaming}_${chat.isProcessComplete}"),
                    chat: chat,
                    onArtifactsButtonPressed: () {
                      if (chat.artifacts.isNotEmpty) {
                        _chatViewModel.downloadArtifact(
                            chat.taskId, chat.artifacts.first.artifactId);
                      } else {
                        Fluttertoast.showToast(
                            msg: "No artifacts for this message.");
                      }
                    },
                  );
                }
              },
            ),
          ),
          if (showGlobalThinkingIndicator)
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: LoadingIndicator(
                isLoading: true,
                statusText: _currentGlobalStatusText,
                thoughtText: _currentGlobalThoughtText,
              ),
            )
          else if (showAgentConnectedMessage)
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Text(
                "✅ Agent Connected. Send a message to start.",
                style: TextStyle(
                    color: Colors.grey[600], fontStyle: FontStyle.italic),
              ),
            )
          else if (chatViewModelReader.isAgentCurrentlyAwaitingUserInput)
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Text(
                "🤔 ${chatViewModelReader.currentAgentQuestionForUser ?? 'Agent is awaiting your input.'}",
                style: TextStyle(
                    color: Colors.orange.shade700,
                    fontWeight: FontWeight.bold),
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ChatInputField(
              viewModel: _chatViewModel,
              onSendPressed: _onSendPressed,
              onContinuousModePressed: () {
                _chatViewModel.isContinuousMode =
                    !_chatViewModel.isContinuousMode;
              },
              isContinuousMode: chatViewModelReader.isContinuousMode,
            ),
          ),
        ],
      ),
    );
  }
}