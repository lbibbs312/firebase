import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/chat/continuous_mode_dialog.dart';
import 'package:flutter/material.dart';
// No need for SharedPreferences import directly here if viewModel handles it.

class ChatInputField extends StatefulWidget {
  // Callback to be triggered when the send button is pressed
  final Function(String) onSendPressed;
  final Function() onContinuousModePressed;
  final bool isContinuousMode;
  // TODO: Create a view model for this class and remove the ChatViewModel
  final ChatViewModel viewModel;

  const ChatInputField({
    Key? key,
    required this.onSendPressed,
    required this.onContinuousModePressed,
    this.isContinuousMode = false,
    required this.viewModel,
  }) : super(key: key);

  @override
  _ChatInputFieldState createState() => _ChatInputFieldState();
}

class _ChatInputFieldState extends State<ChatInputField> {
  // Controller for the TextField to manage its content
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();
  final FocusNode _throwawayFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _focusNode.addListener(() {
      if (_focusNode.hasFocus && widget.isContinuousMode) {
        widget.onContinuousModePressed();
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    _throwawayFocusNode.dispose();
    super.dispose();
  }

  Future<void> _presentContinuousModeDialogIfNeeded() async {
    final showContinuousModeDialog = await widget.viewModel.prefsService
            .getBool('showContinuousModeDialog') ??
        true;

    // Unfocus the text field before showing the dialog to prevent keyboard issues
    FocusScope.of(context).requestFocus(_throwawayFocusNode);

    if (showContinuousModeDialog) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return ContinuousModeDialog(
            onProceed: () {
              Navigator.of(context).pop();
              _executeContinuousMode();
            },
            onCheckboxChanged: (bool value) async {
              await widget.viewModel.prefsService
                  .setBool('showContinuousModeDialog', !value);
            },
          );
        },
      );
    } else {
      _executeContinuousMode();
    }
  }

  void _executeContinuousMode() {
    if (!widget.isContinuousMode) {
      if (_controller.text.isNotEmpty) {
        widget.onSendPressed(_controller.text);
        _controller.clear();
      }
      // Unfocus after sending in non-continuous mode if desired
      // _focusNode.unfocus(); 
    }
    widget.onContinuousModePressed();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // Get the current theme

    return LayoutBuilder(
      builder: (context, constraints) {
        double chatViewWidth = constraints.maxWidth;
        double inputWidth = (chatViewWidth >= 1000) ? 900 : chatViewWidth - 40;

        return Container(
          width: inputWidth,
          constraints: const BoxConstraints(
            minHeight: 50, // Adjust if TextField's intrinsic height is different with theme
            maxHeight: 400,
          ),
          // Removed BoxDecoration to allow TextField to use InputDecorationTheme
          // decoration: BoxDecoration(
          //   color: Colors.white,
          //   border: Border.all(color: Colors.black, width: 0.5),
          //   borderRadius: BorderRadius.circular(8),
          // ),
          // Padding might be handled by InputDecoration.contentPadding from the theme
          // or can be added to InputDecoration if needed.
          // For now, let's rely on the theme's default or TextField's intrinsic padding.
          // padding: const EdgeInsets.symmetric(horizontal: 8), 
          child: SingleChildScrollView( // Handles overflow for multiline input
            reverse: true, // Keeps the latest input visible
            child: TextField(
              controller: _controller,
              focusNode: _focusNode,
              onSubmitted: (text) {
                if (text.isNotEmpty) {
                  widget.onSendPressed(text);
                  _controller.clear();
                }
              },
              maxLines: null, // Allows multiline input
              keyboardType: TextInputType.multiline,
              textInputAction: TextInputAction.send, // Shows send button on keyboard
              style: TextStyle(color: theme.colorScheme.onSurface), // Ensure input text color is correct
              decoration: InputDecoration(
                // Hint text color and style will come from theme.inputDecorationTheme.hintStyle
                hintText: 'Type a message...',
                // Remove internal border, the outer border comes from InputDecorationTheme
                // border: InputBorder.none, (Removed to use theme's borders)
                // contentPadding can be adjusted here if the default is not suitable
                // e.g., contentPadding: EdgeInsets.symmetric(vertical: 10.0, horizontal: 12.0),
                suffixIcon: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (!widget.isContinuousMode)
                      Tooltip(
                        message: 'Send message',
                        child: IconButton(
                          // Icon color will be derived from theme.inputDecorationTheme.suffixIconColor
                          icon: const Icon(Icons.send),
                          onPressed: _controller.text.isEmpty ? null : () {
                            if (_controller.text.isNotEmpty) {
                              widget.onSendPressed(_controller.text);
                              _controller.clear();
                            }
                          },
                          splashRadius: 20, // Standard splash radius
                        ),
                      ),
                    Tooltip(
                      message: widget.isContinuousMode
                          ? 'Pause continuous mode'
                          : 'Enable continuous mode (sends current message if any)',
                      child: IconButton(
                        // Icon color will be derived from theme.inputDecorationTheme.suffixIconColor
                        icon: Icon(widget.isContinuousMode
                            ? Icons.pause_circle_filled_outlined // More distinct pause icon
                            : Icons.play_circle_filled_outlined), // More distinct play icon
                        onPressed: () {
                          if (!widget.isContinuousMode) {
                            _presentContinuousModeDialogIfNeeded();
                          } else {
                            widget.onContinuousModePressed();
                          }
                        },
                        splashRadius: 20, // Standard splash radius
                      ),
                    )
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}