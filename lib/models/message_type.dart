// lib/models/message_type.dart

enum MessageType {
  userMessage,
  loading,
  agentThought,
  agentPlan,
  agentCriticism,
  agentTool,
  actionProposal,
  finalResponse,
  error,
  systemMessage,
  systemError,
  systemInfo,
  agent,
  agentProcess,
  webSearchProgress, // <-- ADD THIS
}