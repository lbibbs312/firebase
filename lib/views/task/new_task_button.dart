// lib/views/task/new_task_button.dart
import 'package:flutter/material.dart';

class NewTaskButton extends StatelessWidget {
  final VoidCallback onPressed;

  const NewTaskButton({Key? key, required this.onPressed}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Determine the width of the TaskView
    double taskViewWidth = MediaQuery.of(context).size.width;
    double buttonWidth = taskViewWidth - 20; // Accounting for padding perhaps
    if (buttonWidth > 260) {
      buttonWidth = 260;
    }

    final theme = Theme.of(context);

    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        // Use theme's card color for background, or surface color
        backgroundColor: theme.cardColor,
        // Use a subtle border color from the theme, or remove if not needed
        side: BorderSide(
            color: theme.colorScheme.onSurface.withOpacity(0.2), width: 0.5),
        // Keep the rounded corners
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8.0),
        ),
        // Ensure elevation is minimal if it's meant to be flat within the panel
        elevation: 1.0,
        // Match padding of other similar elements if necessary
        padding: const EdgeInsets.symmetric(vertical: 0), // Adjust as needed if height is fixed by SizedBox
      ).copyWith(
        // Ensure foreground (text/icon) color contrasts with the new background
        foregroundColor: MaterialStateProperty.all(theme.colorScheme.onSurface),
      ),
      child: SizedBox(
        width: buttonWidth,
        height: 50,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center, // Center content if width is fixed
          children: [
            // Icon color will be inherited from ElevatedButton's foregroundColor
            const Icon(Icons.add),
            const SizedBox(width: 8),
            // Text color will be inherited from ElevatedButton's foregroundColor
            const Text('New Task'),
          ],
        ),
      ),
    );
  }
}