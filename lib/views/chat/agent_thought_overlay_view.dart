// lib/views/chat/agent_thought_overlay_view.dart
import 'dart:async';
// import 'dart:convert'; // JsonEncoder might not be needed if we simplify _formatValue

import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:auto_gpt_flutter_client/views/chat/loading_indicator.dart';

class AnimatedJsonOverlay extends StatefulWidget {
  final Map<String, dynamic> jsonData;
  final List<String> keysToShow;
  final Duration animationInterval;
  final VoidCallback? onAnimationComplete;

  const AnimatedJsonOverlay({
    required this.jsonData,
    this.keysToShow = const ['thoughts', 'plan', 'reasoning', 'self_criticism', 'use_tool'], // Added use_tool
    this.animationInterval = const Duration(seconds: 1), // Faster interval
    this.onAnimationComplete,
    Key? key,
  }) : super(key: key);

  @override
  _AnimatedJsonOverlayState createState() => _AnimatedJsonOverlayState();
}

class _AnimatedJsonOverlayState extends State<AnimatedJsonOverlay> {
  int _currentlyRevealedKeyIndex = -1;
  Timer? _revealTimer;
  List<String> _actualKeysPresent = [];
  bool _isOverlayProcessing = true;

  @override
  void initState() {
    super.initState();
    _actualKeysPresent = widget.keysToShow
        .where((key) =>
            widget.jsonData.containsKey(key) && widget.jsonData[key] != null && _isWorthShowing(widget.jsonData[key]))
        .toList();

    if (_actualKeysPresent.isNotEmpty) {
      _startAnimationSequence();
    } else {
      _isOverlayProcessing = false;
      widget.onAnimationComplete?.call();
    }
  }

  bool _isWorthShowing(dynamic value) {
    if (value == null) return false;
    if (value is String && value.isEmpty) return false;
    if (value is List && value.isEmpty) return false;
    if (value is Map && value.isEmpty) return false;
    // Special check for 'thoughts' if it often contains just empty sub-fields
    if (value is Map && value.keys.length == 1 && value.containsKey('reasoning') && (value['reasoning'] == null || value['reasoning'] == "")) {
        // Example: if thoughts only has an empty reasoning, don't show it.
        // This needs to be tailored to your 'thoughts' structure.
        // For simplicity now, we'll rely on the general empty checks above.
    }
    return true;
  }


  void _startAnimationSequence() {
    _isOverlayProcessing = true;
    _revealTimer = Timer(const Duration(milliseconds: 50), () { // Quicker start
      if (!mounted) return;
      setState(() {
        _currentlyRevealedKeyIndex = 0;
      });
      if (_actualKeysPresent.length > 1) {
        _scheduleNextReveal();
      } else {
        _revealTimer = Timer(widget.animationInterval, _animationDidComplete);
      }
    });
  }

  void _scheduleNextReveal() {
    _revealTimer?.cancel();
    _revealTimer = Timer(widget.animationInterval, () {
      if (!mounted) return;
      if (_currentlyRevealedKeyIndex < _actualKeysPresent.length - 1) {
        setState(() {
          _currentlyRevealedKeyIndex++;
        });
        _scheduleNextReveal();
      } else {
        _animationDidComplete();
      }
    });
  }

  void _animationDidComplete() {
    _revealTimer?.cancel();
    if (mounted) {
      setState(() { _isOverlayProcessing = false; });
    }
    widget.onAnimationComplete?.call();
  }

  @override
  void dispose() {
    _revealTimer?.cancel();
    super.dispose();
  }

  String _formatKeyName(String key) {
    // Convert snake_case to Title Case (e.g., 'use_tool' -> 'Use Tool')
    if (key == 'use_tool' && widget.jsonData[key] is Map) {
      final toolMap = widget.jsonData[key] as Map<String, dynamic>;
      final toolName = toolMap['name'] as String?;
      if (toolName != null) return "Tool: $toolName";
      return "Using Tool";
    }
    return key.replaceAll('_', ' ').split(' ').map((word) => word[0].toUpperCase() + word.substring(1)).join(' ');
  }

  String _formatValue(String key, dynamic value) {
    if (value == null) return 'N/A';

    // Specific formatting for 'use_tool'
    if (key == 'use_tool' && value is Map) {
        final toolMap = value as Map<String, dynamic>;
        final input = toolMap['input'];
        if (input != null && input.toString().isNotEmpty) {
            return 'Input: `${input.toString().replaceAll('`', '\\`')}`'; // Simple markdown for input
        }
        return "Executing with provided parameters."; // Fallback if no input or complex args
    }

    if (value is String) {
      if ((value.startsWith('{') && value.endsWith('}')) || (value.startsWith('[') && value.endsWith(']'))) {
        return '```json\n$value\n```';
      }
      return value;
    }
    if (value is Map || value is List) {
      // Simplified representation for Map/List to avoid compiler stress from JsonEncoder
      // Wrapped in markdown code block for better display.
      String simpleString = value.toString();
      if (simpleString.length > 300) { // Truncate very long simple strings
          simpleString = simpleString.substring(0, 300) + "... (truncated)";
      }
      return '```\n$simpleString\n```';
    }
    return value.toString();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDarkTheme = theme.brightness == Brightness.dark;

    if (_actualKeysPresent.isEmpty) { // Don't show anything if no relevant keys
      return const SizedBox.shrink();
    }

    Color cardBackgroundColor = isDarkTheme ? AppColors.codeBlockArtefactsDark.withOpacity(0.85) : AppColors.codeBlockArtefactsLight.withOpacity(0.85);
    Color textColor = isDarkTheme ? AppColors.neutral1Dark : AppColors.neutral2Light;
    Color keyColor = isDarkTheme ? AppColors.ultraWhiteDark.withOpacity(0.9) : AppColors.ultraWhiteLight.withOpacity(0.9);
    Color accentColor = isDarkTheme ? AppColors.primaryDark.withOpacity(0.7) : AppColors.primaryLight.withOpacity(0.7);

    String statusTextForLoadingIndicator = "";
     if (_isOverlayProcessing && _currentlyRevealedKeyIndex >= 0 && _currentlyRevealedKeyIndex < _actualKeysPresent.length) {
        statusTextForLoadingIndicator = "Processing: ${_formatKeyName(_actualKeysPresent[_currentlyRevealedKeyIndex])}...";
     } else if (_isOverlayProcessing) {
        statusTextForLoadingIndicator = "Agent is processing thoughts...";
     }


    return Card(
      elevation: 1,
      margin: const EdgeInsets.only(bottom: 0.0), // Tighter margin if it's part of a larger bubble
      color: cardBackgroundColor,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_isOverlayProcessing && _actualKeysPresent.isNotEmpty) // Show spinner only if animating content
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: LoadingIndicator( // Minimalist spinner for overlay
                  isLoading: true, // Always true while overlay is processing
                  statusText: statusTextForLoadingIndicator,
                ),
              ),
            
            for (int i = 0; i < _actualKeysPresent.length; i++)
              AnimatedOpacity(
                opacity: i <= _currentlyRevealedKeyIndex ? 1.0 : 0.0,
                duration: const Duration(milliseconds: 500), // Slightly slower fade for readability
                curve: Curves.easeOut,
                child: i <= _currentlyRevealedKeyIndex
                    ? Padding(
                        padding: const EdgeInsets.only(bottom: 8.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "${_formatKeyName(_actualKeysPresent[i])}:",
                              style: TextStyle(
                                // fontFamily: 'monospace',
                                fontSize: 13,
                                fontWeight: FontWeight.w600, // Bolder key
                                color: keyColor,
                              ),
                            ),
                            const SizedBox(height: 3),
                            Container(
                              padding: const EdgeInsets.only(left: 10, top: 4, bottom: 4, right: 4),
                              decoration: BoxDecoration(
                                border: Border(
                                  left: BorderSide(color: accentColor, width: 2.5)
                                )
                              ),
                              child: MarkdownBody(
                                data: _formatValue(_actualKeysPresent[i], widget.jsonData[_actualKeysPresent[i]]) ?? '...',
                                selectable: true,
                                styleSheet: MarkdownStyleSheet.fromTheme(theme).copyWith(
                                  p: TextStyle(fontSize: 12, color: textColor, height: 1.45),
                                  code: TextStyle(
                                    fontFamily: 'monospace',
                                    fontSize: 11,
                                    color: textColor.withOpacity(0.9),
                                    backgroundColor: (isDarkTheme ? AppColors.chatBackgroundDark : AppColors.chatBackgroundLight).withOpacity(0.4)
                                  ),
                                  codeblockPadding: const EdgeInsets.all(8),
                                  codeblockDecoration: BoxDecoration(
                                    color: (isDarkTheme ? AppColors.chatBackgroundDark : AppColors.chatBackgroundLight).withOpacity(0.4),
                                    borderRadius: BorderRadius.circular(4)
                                  )
                                ),
                              ),
                            ),
                          ],
                        ),
                      )
                    : const SizedBox.shrink(),
              ),
          ],
        ),
      ),
    );
  }
}