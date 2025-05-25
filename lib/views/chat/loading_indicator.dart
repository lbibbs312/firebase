import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:flutter/material.dart';

class LoadingIndicator extends StatefulWidget {
  final bool isLoading; // Controls the animated bar visibility and animation
  final String? statusText;
  final String? thoughtText;

  const LoadingIndicator({
    Key? key,
    required this.isLoading,
    this.statusText,
    this.thoughtText,
  }) : super(key: key);

  @override
  _LoadingIndicatorState createState() => _LoadingIndicatorState();
}

class _LoadingIndicatorState extends State<LoadingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    if (widget.isLoading) {
      _controller.repeat();
    }
  }

  @override
  void didUpdateWidget(LoadingIndicator oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isLoading && !_controller.isAnimating) {
      _controller.repeat();
    } else if (!widget.isLoading && _controller.isAnimating) {
      _controller.stop();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        double width =
            (constraints.maxWidth >= 1000) ? 850 : constraints.maxWidth - 65;

        return Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(
              width: width,
              height: 4.0,
              child: widget.isLoading
                  ? AnimatedBuilder(
                      animation: _controller,
                      builder: (context, child) {
                        return ShaderMask(
                          shaderCallback: (rect) {
                            return LinearGradient(
                              begin: Alignment.centerLeft,
                              end: Alignment.centerRight,
                              colors: [
                                Colors.grey[400]!,
                                AppColors.primaryLight,
                                Colors.white,
                                Colors.grey[400]!,
                              ],
                              stops: [
                                _controller.value - 0.5,
                                _controller.value - 0.25,
                                _controller.value,
                                _controller.value + 0.25,
                              ],
                            ).createShader(rect);
                          },
                          child: Container(
                            width: width,
                            height: 4.0,
                            color: Colors.white, // Background for the shader
                          ),
                        );
                      },
                    )
                  : Container(
                      width: width,
                      height: 4.0,
                      color: Colors.grey[400], // Idle state
                    ),
            ),
            // These will only build if statusText/thoughtText are provided and not empty
            if (widget.statusText != null && widget.statusText!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6.0),
                child: Text(
                  widget.statusText!,
                  style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                  textAlign: TextAlign.center,
                ),
              ),
            if (widget.thoughtText != null && widget.thoughtText!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4.0),
                child: Container(
                  width: width * 0.9, // Slightly narrower than the bar
                  padding: const EdgeInsets.all(8.0),
                  decoration: BoxDecoration(
                    color: Colors.blueGrey[50],
                    borderRadius: BorderRadius.circular(4.0),
                    border: Border.all(color: Colors.blueGrey[100]!)
                  ),
                  child: Text(
                    "Agent thoughts: ${widget.thoughtText!}",
                    style: TextStyle(
                        fontSize: 12,
                        fontStyle: FontStyle.italic,
                        color: Colors.blueGrey[700]),
                  ),
                ),
              ),
          ],
        );
      },
    );
  }
}