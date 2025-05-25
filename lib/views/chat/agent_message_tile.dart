// lib/views/chat/agent_message_tile.dart
import 'dart:convert'; // For JsonEncoder
import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // For Clipboard
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:fluttertoast/fluttertoast.dart'; // For toast messages
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:url_launcher/url_launcher.dart'; // For web fallback

// Import webview_flutter. This might still be an issue if the package
// itself has top-level web-incompatible code or build steps.
// If this import line *itself* breaks the web build, the single-file approach
// without stubs is not feasible for functional WebView on mobile.
import 'package:webview_flutter/webview_flutter.dart';


class AgentMessageTile extends StatefulWidget {
  final Chat chat;
  final VoidCallback? onArtifactsButtonPressed;

  const AgentMessageTile({
    Key? key,
    required this.chat,
    this.onArtifactsButtonPressed,
  }) : super(key: key);

  @override
  _AgentMessageTileState createState() => _AgentMessageTileState();
}

class _AgentMessageTileState extends State<AgentMessageTile>
    with TickerProviderStateMixin {
  bool _isExecutionStepsExpanded = false;

  bool _showProcessThoughts = false;
  bool _showProcessPlan = false;
  bool _showProcessCriticism = false;
  bool _showProcessTool = false;

  late AnimationController _tileFadeController;
  late Animation<double> _tileFadeAnimation;

  // Controller for the WebView, only initialized and used if !kIsWeb
  WebViewController? _mobileWebViewController;

  @override
  void initState() {
    super.initState();

    _tileFadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 450),
    );
    _tileFadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _tileFadeController, curve: Curves.easeOutCubic),
    );
    _tileFadeController.forward();

    if (widget.chat.messageType == MessageType.agentProcess && !widget.chat.isProcessComplete) {
      _resetInternalAnimationFlags();
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _triggerSequentialInternalAnimations();
      });
    }
  }

  @override
  void didUpdateWidget(covariant AgentMessageTile oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.chat.id != oldWidget.chat.id) {
      _tileFadeController.reset();
      _tileFadeController.forward();
      _isExecutionStepsExpanded = false;
      if (widget.chat.messageType == MessageType.agentProcess && !widget.chat.isProcessComplete) {
        _resetInternalAnimationFlags();
         WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) _triggerSequentialInternalAnimations();
        });
      }
    } else if (widget.chat.messageType == MessageType.agentProcess &&
               !widget.chat.isProcessComplete &&
               widget.chat.currentThoughtPlanData != oldWidget.chat.currentThoughtPlanData) {
      _resetInternalAnimationFlags();
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _triggerSequentialInternalAnimations();
      });
    }
  }

  @override
  void dispose() {
    _tileFadeController.dispose();
    super.dispose();
  }

  void _resetInternalAnimationFlags() {
    _showProcessThoughts = false;
    _showProcessPlan = false;
    _showProcessCriticism = false;
    _showProcessTool = false;
  }

  void _triggerSequentialInternalAnimations() async {
    // ... (this method remains unchanged)
    if (!mounted || widget.chat.messageType != MessageType.agentProcess) return;
    
    const Duration initialBubbleSettleDelay = Duration(milliseconds: 300);
    const Duration itemRevealDelay = Duration(milliseconds: 800); 

    await Future.delayed(initialBubbleSettleDelay);

    final data = widget.chat.currentThoughtPlanData;
    final thoughtsData = data?['thoughts'];

    if (thoughtsData is Map && thoughtsData['text'] != null && (thoughtsData['text'] as String).isNotEmpty) {
      if (mounted) setState(() => _showProcessThoughts = true);
      await Future.delayed(itemRevealDelay);
    } else {
      if (mounted) setState(() => _showProcessThoughts = false);
    }
    if (!mounted) return;

    if (data?['plan'] != null && data!['plan'] is List && (data['plan'] as List).isNotEmpty) {
      if (mounted) setState(() => _showProcessPlan = true);
      await Future.delayed(itemRevealDelay);
    } else {
      if (mounted) setState(() => _showProcessPlan = false);
    }
    if (!mounted) return;

    if (thoughtsData is Map && thoughtsData['self_criticism'] != null && (thoughtsData['self_criticism'] as String).isNotEmpty) {
      if (mounted) setState(() => _showProcessCriticism = true);
      await Future.delayed(itemRevealDelay);
    } else {
      if (mounted) setState(() => _showProcessCriticism = false);
    }
    if (!mounted) return;

    if (data?['use_tool']?['name'] != null) {
      if (mounted) setState(() => _showProcessTool = true);
    } else {
      if (mounted) setState(() => _showProcessTool = false);
    }
  }

  bool _containsMarkdown(String text) {
    // ... (this method remains unchanged)
    return text.contains('```') || text.contains('* ') || text.contains('\n- ') || text.contains('##');
  }

  Widget _buildMainMessageContent(BuildContext context, String message,
      Color textColor, bool isDarkTheme,
      {required bool isStatus, required bool isStreaming, bool applyGlow = false}) {
    // ... (this method remains unchanged)
    final theme = Theme.of(context);
    TextStyle baseStyle = isStatus
        ? theme.textTheme.bodyMedium!.copyWith(color: textColor.withOpacity(0.85), fontStyle: FontStyle.italic, fontSize: 13)
        : theme.textTheme.bodyLarge!.copyWith(color: textColor, height: 1.45, fontSize: 15);

    if (applyGlow) {
      baseStyle = baseStyle.copyWith(
        shadows: [
          Shadow(
            blurRadius: isDarkTheme ? 10.0 : 8.0,
            color: (isDarkTheme ? Colors.lightBlue.shade200 : Colors.blue.shade300).withOpacity(0.55),
            offset: const Offset(0, 0),
          ),
           Shadow(
            blurRadius: 16.0,
            color: (isDarkTheme ? Colors.cyanAccent : Colors.lightBlueAccent).withOpacity(0.25),
            offset: const Offset(0, 0),
          ),
        ],
      );
    }
    
    if (message.isEmpty && isStreaming && widget.chat.messageType != MessageType.agentProcess) {
      return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (i) => _Dot(delay: i * 200, size: 3.0, color: textColor.withOpacity(0.5))));
    } else if (_containsMarkdown(message) && !isStatus) {
      return MarkdownBody(
        data: message,
        selectable: true,
        styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
          p: baseStyle,
          code: baseStyle.copyWith(fontFamily: 'monospace', backgroundColor: isDarkTheme ? Colors.black.withOpacity(0.3) : Colors.grey.shade200, color: isDarkTheme ? Colors.lightGreenAccent.shade100 : Colors.indigo.shade900),
          codeblockDecoration: BoxDecoration(
              color: isDarkTheme ? Colors.grey.shade800.withOpacity(0.5) : Colors.grey.shade100,
              borderRadius: BorderRadius.circular(6.0),
              border: Border.all(color: (isDarkTheme ? Colors.grey.shade700 : Colors.grey.shade300).withOpacity(0.7))
          ),
          h1: baseStyle.copyWith(fontSize: baseStyle.fontSize! * 1.4, fontWeight: FontWeight.bold),
          h2: baseStyle.copyWith(fontSize: baseStyle.fontSize! * 1.3, fontWeight: FontWeight.bold),
          h3: baseStyle.copyWith(fontSize: baseStyle.fontSize! * 1.2, fontWeight: FontWeight.bold),
        ),
      );
    } else {
      return SelectableText(message, style: baseStyle);
    }
  }

  Widget _buildProcessDetailItem(BuildContext context, String label, dynamic content, bool isVisible, {bool isList = false, bool isCode = false, bool applyGlowToContent = false}) {
    // ... (this method remains unchanged)
    final theme = Theme.of(context);
    final bool isDarkThemeLocal = theme.brightness == Brightness.dark;
    Color detailLabelColor = (isDarkThemeLocal ? AppColors.neutral1Dark : AppColors.neutral2Light).withOpacity(0.75);
    Color currentDetailContentColor = isDarkThemeLocal ? AppColors.neutral1Dark.withOpacity(0.95) : AppColors.neutral2Light.withOpacity(0.95);

    TextStyle contentStyle = TextStyle(color: currentDetailContentColor, fontSize: 13, height: 1.4);
    if (applyGlowToContent) {
        contentStyle = contentStyle.copyWith(shadows: [
            Shadow(blurRadius: 3.0, color: Colors.blue.withOpacity(0.25), offset: const Offset(0,0))
        ]);
    }

    if (content == null || (content is String && content.isEmpty) || (content is List && content.isEmpty)) {
      return const SizedBox.shrink();
    }

    Widget contentWidget;
    if (isList && content is List) {
      contentWidget = Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: content.map<Widget>((item) => Padding(
          padding: const EdgeInsets.only(top: 3.0, left: 0),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.only(right: 6.0, top: 2.0),
                child: Text("•", style: contentStyle.copyWith(color: detailLabelColor, fontWeight: FontWeight.bold)),
              ),
              Expanded(child: SelectableText(item.toString(), style: contentStyle)),
            ],
          ),
        )).toList(),
      );
    } else if (isCode && content is String) {
        contentWidget = Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          margin: const EdgeInsets.only(top: 2),
          decoration: BoxDecoration(
            color: isDarkThemeLocal ? Colors.black.withOpacity(0.35) : Colors.grey.shade100,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: (isDarkThemeLocal ? Colors.grey.shade700 : Colors.grey.shade300).withOpacity(0.5))
          ),
          child: SelectableText(content.toString(), style: TextStyle(fontFamily: 'monospace', color: currentDetailContentColor, fontSize: 12.5).merge(applyGlowToContent ? TextStyle(shadows: contentStyle.shadows) : null)),
        );
    }
     else {
      contentWidget = _buildMainMessageContent(context, content.toString(), currentDetailContentColor, isDarkThemeLocal, isStatus: false, isStreaming: false, applyGlow: applyGlowToContent);
    }

    return AnimatedOpacity(
      opacity: isVisible ? 1.0 : 0.0,
      duration: const Duration(milliseconds: 550),
      curve: Curves.easeOut,
      child: Visibility( 
        visible: isVisible,
        maintainState: true, maintainAnimation: true, maintainSize: false,
        child: Padding(
          padding: const EdgeInsets.only(top: 14.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("$label:", style: TextStyle(color: detailLabelColor, fontWeight: FontWeight.w600, fontSize: 13)),
              const SizedBox(height: 6),
              Padding(
                padding: const EdgeInsets.only(left: 4.0),
                child: contentWidget,
              ),
            ],
          ),
        ),
      ),
    );
  }

// In AgentMessageTile.dart
Widget _buildWebViewPlaceholder(BuildContext context, String searchUrl, String query, bool isDarkTheme) {
   // searchUrl here is the google.com/search?q=...
   // query is the actual search term
   return Container(
    key: ValueKey("webview_placeholder_${widget.chat.id}"),
    padding: const EdgeInsets.all(12.0),
    decoration: BoxDecoration(
      color: isDarkTheme ? Colors.grey[800] : Colors.grey[300],
      borderRadius: BorderRadius.circular(10.0),
    ),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.search, size: 18, color: Theme.of(context).textTheme.bodySmall?.color),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                'Web Search: "$query"',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          'Content preview is not shown here on the web. Click below to see results.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          icon: Icon(Icons.open_in_new, size: 16),
          label: Text('View on Google', style: TextStyle(fontSize: 12)),
          onPressed: () async {
            final uri = Uri.parse(searchUrl); // Use the direct Google search URL
            if (await canLaunchUrl(uri)) {
              await launchUrl(uri, webOnlyWindowName: '_blank');
            } else {
               if (mounted) { // Ensure widget is still mounted
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Could not launch $searchUrl')),
                  );
               }
            }
          },
          style: ElevatedButton.styleFrom(
            padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            backgroundColor: Theme.of(context).colorScheme.primary,
            foregroundColor: Theme.of(context).colorScheme.onPrimary,
          ),
        ),
      ],
    ),
  );
}

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bool isDarkTheme = theme.brightness == Brightness.dark;
    final Chat chat = widget.chat;
    Widget tileContent; 

    if (chat.messageType == MessageType.agentProcess) {
      // ... (agentProcess logic - unchanged) ...
      Color processTextColor = isDarkTheme ? AppColors.neutral1Dark : AppColors.neutral2Light;
      Color processIconColor = processTextColor.withOpacity(0.7);
      Color processCardBg = isDarkTheme ? const Color(0xFF2C2C2E) : const Color(0xFFF2F2F7);
      String headerText = chat.message.isNotEmpty ? chat.message : (chat.processSteps != null && chat.processSteps!.isNotEmpty ? chat.processSteps!.last : "Agent analyzing...");
      final thoughtsDataMap = chat.currentThoughtPlanData?['thoughts'];
      String? thoughtsText = thoughtsDataMap is Map ? thoughtsDataMap['text'] as String? : null;
      String? criticismText = thoughtsDataMap is Map ? thoughtsDataMap['self_criticism'] as String? : null;
      List<dynamic>? planItemsDynamic = chat.currentThoughtPlanData?['plan'] as List<dynamic>?;
      List<String>? planItems = planItemsDynamic?.map((item) => item.toString()).toList();
      final toolDataMap = chat.currentThoughtPlanData?['use_tool'];
      String? toolNameInProcess = toolDataMap is Map ? toolDataMap['name'] as String? : null;
      Map<String,dynamic>? toolArgsMap = toolDataMap is Map ? (toolDataMap['arguments'] is Map ? Map<String,dynamic>.from(toolDataMap['arguments']) : null) : null;
      String? toolArgsString;
      if (toolArgsMap != null && toolArgsMap.isNotEmpty) {
          try { toolArgsString = JsonEncoder.withIndent('  ').convert(toolArgsMap); } 
          catch (e) { toolArgsString = toolArgsMap.toString(); }
      }

      tileContent = Container(
        margin: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
        padding: const EdgeInsets.fromLTRB(0, 0, 0, 8),
        decoration: BoxDecoration(
            color: processCardBg,
            borderRadius: BorderRadius.circular(14.0),
            boxShadow: [ BoxShadow( color: Colors.black.withOpacity(isDarkTheme ? 0.2 : 0.08), blurRadius: 10, spreadRadius: -2, offset: const Offset(0, 4),) ]),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
             Padding(
              padding: const EdgeInsets.fromLTRB(16.0, 14.0, 16.0, 10.0),
               child: Row(
                children: [
                  Icon(Icons.psychology_alt_rounded, size: 20, color: processIconColor),
                  const SizedBox(width: 10),
                  Expanded(child: Text(headerText, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: processTextColor.withOpacity(0.95)), maxLines: 1, overflow: TextOverflow.ellipsis)),
                ],
              ),
            ),
            if (chat.processSteps != null && chat.processSteps!.isNotEmpty)
              Theme( data: theme.copyWith(dividerColor: Colors.transparent), child: ExpansionTile( key: PageStorageKey<String>("${chat.id}_process_steps_expansion_tile"), iconColor: processIconColor.withOpacity(0.7), collapsedIconColor: processIconColor.withOpacity(0.7), tilePadding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 0.0), title: Text( "Execution Steps (${chat.processSteps!.length})", style: TextStyle(fontSize: 12, color: processTextColor.withOpacity(0.8), fontWeight: FontWeight.w500),), initiallyExpanded: _isExecutionStepsExpanded, onExpansionChanged: (bool expanded) { setState(() { _isExecutionStepsExpanded = expanded; }); }, 
              children: <Widget>[
                  Padding(
                    padding: const EdgeInsets.only(left: 18.0, right: 18.0, bottom: 10.0, top: 0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: chat.processSteps!.map((step) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 3.0),
                        child: Text("• $step", style: TextStyle(fontSize: 12, color: processTextColor.withOpacity(0.75), height: 1.3)),
                      )).toList(),
                    ),
                  )
              ],),),
            if (thoughtsText != null || planItems != null || criticismText != null || toolNameInProcess != null)
              Padding(
                padding: EdgeInsets.fromLTRB( 16.0, (chat.processSteps != null && chat.processSteps!.isNotEmpty && !_isExecutionStepsExpanded) ? 4.0 : ((chat.processSteps == null || chat.processSteps!.isEmpty) ? 0 : 8.0), 16.0, 10.0 ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (thoughtsText != null && thoughtsText.isNotEmpty)
                      _buildProcessDetailItem(context, "Thoughts", thoughtsText, _showProcessThoughts, applyGlowToContent: true),
                    if (planItems != null && planItems.isNotEmpty)
                      _buildProcessDetailItem(context, "Plan", planItems, _showProcessPlan, isList: true, applyGlowToContent: false),
                    if (criticismText != null && criticismText.isNotEmpty)
                      _buildProcessDetailItem(context, "Criticism", criticismText, _showProcessCriticism, applyGlowToContent: true),
                    if (toolNameInProcess != null)
                      _buildProcessDetailItem(context, "Tool Activity", "$toolNameInProcess${toolArgsString != null ? '\nArguments:\n$toolArgsString' : ''}", _showProcessTool, isCode: toolArgsString != null, applyGlowToContent: false ),
                  ],),),
            if (chat.isStreaming && chat.messageType == MessageType.agentProcess && !chat.isProcessComplete)
              Padding( padding: const EdgeInsets.only(top: 4.0, left: 16.0, bottom: 8.0), child: Row(mainAxisSize: MainAxisSize.min, children: List.generate(3, (i) => _Dot(delay: i * 200, color: processTextColor))),)
          ],),);
    } 
    else if (chat.messageType == MessageType.agentTool) {
      final String? toolNameFromResponse = chat.jsonResponse?['tool_name'] as String?;
      final args = chat.jsonResponse?['arguments'] as Map<String, dynamic>?;
      String toolDisplayMessage = chat.message; 
      Widget? toolSpecificContent;

      if ((toolNameFromResponse == 'web_search' || toolNameFromResponse == 'search') && args?['query'] is String) {
        final query = args!['query'] as String;
        final url = 'https://www.google.com/search?q=${Uri.encodeComponent(query)}';
        
        toolDisplayMessage = "Searching for: \"$query\"";

        if (kIsWeb) {
          toolSpecificContent = _buildWebViewPlaceholder(context, url, query, isDarkTheme);
        } else {
          // Initialize _mobileWebViewController only when needed and if not already initialized
          // for this specific URL. For simplicity, we might re-initialize.
          // A more complex state management might be needed if you want to preserve WebView state across rebuilds.
          _mobileWebViewController = WebViewController()
            ..setJavaScriptMode(JavaScriptMode.unrestricted)
            ..setBackgroundColor(isDarkTheme ? const Color(0xFF1E1E1E) : const Color(0xFFF0F0F0))
            // Add NavigationDelegate if needed for loading states, errors etc.
            ..loadRequest(Uri.parse(url));

          toolSpecificContent = ClipRRect(
            borderRadius: BorderRadius.circular(10.0),
            child: SizedBox(
              height: 350, 
              child: WebViewWidget(controller: _mobileWebViewController!), // Assert non-null here
            ),
          );
        }
      }

      tileContent = Container(
        margin: const EdgeInsets.only(left: 16.0, right: 16.0, top: 8.0, bottom: 8.0),
        padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 12.0),
        decoration: BoxDecoration(
          color: isDarkTheme ? Colors.grey[850]!.withOpacity(0.7) : Colors.grey[200],
          borderRadius: BorderRadius.circular(14.0),
          border: Border.all(color: (isDarkTheme ? Colors.grey.shade700 : Colors.grey.shade300).withOpacity(0.5))
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
               children: [
                Icon(Icons.construction_rounded, size: 18, color: (isDarkTheme ? AppColors.neutral1Dark : AppColors.neutral2Light).withOpacity(0.7)),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    (toolNameFromResponse == 'web_search' && args?['query'] is String) ? "Web Search" : (toolNameFromResponse ?? "Tool Activity"),
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: (isDarkTheme ? AppColors.neutral1Dark : AppColors.neutral2Light).withOpacity(0.8)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
             _buildMainMessageContent(
                context, 
                toolDisplayMessage,
                isDarkTheme ? AppColors.neutral1Dark : AppColors.neutral2Light, 
                isDarkTheme,
                applyGlow: false,
                isStatus: true, 
                isStreaming: false, 
              ),
            if (toolSpecificContent != null) 
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: toolSpecificContent,
              ),
          ],
        ),
      );
    }
    // ... (rest of the build method for other message types - unchanged) ...
    else if (chat.messageType == MessageType.agentThought ||
               chat.messageType == MessageType.agentPlan ||
               chat.messageType == MessageType.agentCriticism) {
      String label = ""; IconData? prefixIconData;
      Color textColor = isDarkTheme ? AppColors.neutral1Dark.withOpacity(0.85) : AppColors.neutral2Light.withOpacity(0.85);
      switch (chat.messageType) {
        case MessageType.agentThought: label = "Thought"; prefixIconData = Icons.bubble_chart_outlined; break;
        case MessageType.agentPlan: label = "Plan"; prefixIconData = Icons.checklist_rtl_rounded; break;
        case MessageType.agentCriticism: label = "Criticism"; prefixIconData = Icons.rate_review_outlined; break;
        default: break;
      }
      tileContent = Container(
        margin: const EdgeInsets.only(left: 16.0, right: 50.0, top: 5.0, bottom: 5.0),
        padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 10.0),
        decoration: BoxDecoration( color: isDarkTheme ? const Color(0xFF2A2A2C) : const Color(0xFFE5E5EA), borderRadius: BorderRadius.circular(14.0), border: Border.all(color: (isDarkTheme ? Colors.grey.shade700 : Colors.grey.shade300).withOpacity(0.6))),
        child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (prefixIconData != null) 
                Padding(
                  padding: const EdgeInsets.only(right: 10.0, top: 1.0),
                  child: Icon(prefixIconData, size: 18, color: textColor.withOpacity(0.7)),
                ),
              Expanded(child: _buildMainMessageContent(context, chat.message, textColor, isDarkTheme, isStatus: true, isStreaming: false)),
            ],
          ),
      );
    } 
    else { 
      bool isSimpleStatus = chat.messageType == MessageType.loading || chat.messageType == MessageType.systemMessage;
      Color mainContentTextColor = isDarkTheme ? const Color(0xFFE1E1E3) : const Color(0xFF1D1D1F);

      if (isSimpleStatus) {
        tileContent = Container(
          alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
          child: _buildMainMessageContent(context, chat.message, mainContentTextColor.withOpacity(0.7), isDarkTheme, isStatus: true, isStreaming: chat.isStreaming),
        );
      } else if (chat.messageType == MessageType.error) {
        tileContent = Container(
          margin: const EdgeInsets.symmetric(vertical: 6.0, horizontal: 16.0),
          padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 10.0),
          decoration: BoxDecoration(
            color: (isDarkTheme ? Colors.red.shade900 : Colors.red.shade100).withOpacity(0.6),
            borderRadius: BorderRadius.circular(12.0),
             border: Border.all(color: (isDarkTheme ? Colors.red.shade700 : Colors.red.shade300).withOpacity(0.5))
          ),
          child: Row(
            children: [
              Icon(Icons.error_outline_rounded, color: isDarkTheme ? Colors.red.shade200 : Colors.red.shade700, size: 20),
              const SizedBox(width: 10),
              Expanded(child: SelectableText(chat.message, style: TextStyle(color: isDarkTheme ? Colors.red.shade100 : Colors.red.shade900, fontSize: 14))),
            ],
          ),
        );
      } else { 
        tileContent = Padding(
          padding: const EdgeInsets.only(left: 16.0, right: 16.0, top: 8.0, bottom: 10.0),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildMainMessageContent(
                        context, chat.message, mainContentTextColor, isDarkTheme,
                        applyGlow: true, isStreaming: chat.isStreaming, isStatus: false, 
                    ),
                    if (!chat.isStreaming && chat.messageType != MessageType.error)
                      Padding(
                        padding: const EdgeInsets.only(top: 10.0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.start,
                          children: [
                            _MessageActionButton(icon: Icons.copy_rounded, tooltip: "Copy message", onPressed: () { Clipboard.setData(ClipboardData(text: chat.message)); Fluttertoast.showToast(msg: "Message copied!"); }),
                            if (widget.onArtifactsButtonPressed != null && chat.artifacts.isNotEmpty)
                              _MessageActionButton(icon: Icons.attach_file_rounded, tooltip: "View Artifacts (${chat.artifacts.length})", onPressed: widget.onArtifactsButtonPressed!),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
        );
      }
    }

    return FadeTransition(
      opacity: _tileFadeAnimation,
      child: AnimatedSize(
        duration: const Duration(milliseconds: 250), 
        curve: Curves.fastOutSlowIn,
        alignment: Alignment.topCenter,
        child: tileContent,
      ),
    );
  }
}

// ... (_MessageActionButton and _Dot classes remain unchanged) ...
class _MessageActionButton extends StatelessWidget {
  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;

  const _MessageActionButton({
    Key? key,
    required this.icon,
    required this.tooltip,
    required this.onPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDarkTheme = theme.brightness == Brightness.dark;
    Color actionIconColor = (isDarkTheme ? AppColors.neutral1Dark : AppColors.neutral2Light).withOpacity(0.65);

    return Tooltip(
      message: tooltip,
      preferBelow: true,
      waitDuration: const Duration(milliseconds: 400),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(16.0),
          splashColor: Theme.of(context).splashColor.withOpacity(0.1),
          highlightColor: Theme.of(context).highlightColor.withOpacity(0.05),
          child: Padding(
            padding: const EdgeInsets.all(6.0),
            child: Icon(icon, size: 18, color: actionIconColor),
          ),
        ),
      ),
    );
  }
}

class _Dot extends StatefulWidget {
  final int delay;
  final double size;
  final Color color;
  const _Dot({required this.delay, this.size = 3.0, this.color = Colors.grey, Key? key}) : super(key: key);
  @override _DotState createState() => _DotState();
}

class _DotState extends State<_Dot> with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 700))
      ..repeat(reverse: true);
  }

  @override
  Widget build(BuildContext context) {
    final CurvedAnimation curvedAnimation = CurvedAnimation(
        parent: _ctrl,
        curve: Interval(
          (widget.delay % 600) / 600.0,
          ((widget.delay % 600) + 350) / 600.0,
          curve: Curves.easeInOut,
        ));
    return FadeTransition(
      opacity: Tween(begin: 0.4, end: 0.9).animate(curvedAnimation),
      child: ScaleTransition(
        scale: Tween(begin: 0.8, end: 1.0).animate(curvedAnimation),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 2.0),
          child: CircleAvatar(radius: widget.size, backgroundColor: widget.color.withOpacity(0.7)),
        ),
      ),
    );
  }
  @override void dispose() { _ctrl.dispose(); super.dispose(); }
}