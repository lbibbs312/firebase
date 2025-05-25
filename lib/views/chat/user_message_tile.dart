import 'package:flutter/material.dart';

class UserMessageTile extends StatelessWidget {
  final String message;

  // Constructor takes the user message as a required parameter
  const UserMessageTile({
    Key? key,
    required this.message,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // Get the current theme

    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate the width of the chat view based on the constraints provided
        double chatViewWidth = constraints.maxWidth;

        // Determine the width of the message tile based on the chat view width
        double tileWidth = (chatViewWidth >= 1000) ? 900 : chatViewWidth - 40;

        return Align(
          alignment: Alignment.center, // Or Alignment.centerRight for user messages, Alignment.centerLeft for agent
          child: Container(
            width: tileWidth,
            // Minimum height constraint for the container
            constraints: const BoxConstraints(
              minHeight: 50, // Ensure this accommodates padding and text
            ),
            // Margin and padding for styling
            margin: const EdgeInsets.symmetric(vertical: 8),
            // Using EdgeInsets.all to ensure consistent padding around the content
            padding: const EdgeInsets.all(12.0), // Adjusted padding for better spacing
            // Decoration to style the container
            decoration: BoxDecoration(
              // Use theme's cardColor or surface for the background
              color: theme.cardColor, 
              // Use a subtle border from the theme
              border: Border.all(
                  color: theme.colorScheme.onSurface.withOpacity(0.15),
                  width: 1),
              // Consistent rounded corners
              borderRadius: BorderRadius.circular(8), 
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start, // Align "User" label with top of message
              children: [
                // "User" label with custom styling
                Text(
                  "User",
                  style: TextStyle(
                    // Use theme's onSurface color for text
                    color: theme.colorScheme.onSurface, 
                    fontSize: 15, // Slightly adjusted for balance
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 16), // Adjusted spacing
                // Expanded widget to accommodate the message text
                Expanded(
                  child: SelectableText(
                    message,
                    style: TextStyle(
                      // Use theme's onSurface color for message text
                      color: theme.colorScheme.onSurface.withOpacity(0.85), 
                      fontSize: 14,
                      height: 1.4, // Improve line spacing for readability
                    ),
                    maxLines: null,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}