import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';

class TaskListTile extends StatelessWidget {
  final Task task;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  final bool selected;

  const TaskListTile({
    Key? key,
    required this.task,
    required this.onTap,
    required this.onDelete,
    this.selected = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // Define colors based on selection state and theme
    Color backgroundColor;
    Color iconColor;
    Color textColor;
    Color closeButtonColor; // Non-nullable
    Color borderColor;

    if (selected) {
      // For selected items, use primary color scheme
      backgroundColor = theme.colorScheme.primary;
      iconColor = theme.colorScheme.onPrimary;
      textColor = theme.colorScheme.onPrimary;
      closeButtonColor = theme.colorScheme.onPrimary; // Assigned for selected state
      borderColor = theme.colorScheme.primary;
    } else {
      // For unselected items, use cardColor and onSurface colors
      backgroundColor = theme.cardColor;
      iconColor = theme.colorScheme.onSurface.withOpacity(0.7);
      textColor = theme.colorScheme.onSurface;
      // FIX: Initialize closeButtonColor in the else branch as well to satisfy null safety.
      // This color won't actually be used if selected is false, as the button isn't rendered.
      closeButtonColor = theme.colorScheme.onSurface; 
      borderColor = theme.colorScheme.onSurface.withOpacity(0.15);
    }

    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 52, // Consistent height for items
        margin: const EdgeInsets.symmetric(horizontal: 10.0, vertical: 4.0), // Margin between items
        padding: const EdgeInsets.symmetric(horizontal: 16.0), // Internal padding
        decoration: BoxDecoration(
          color: backgroundColor,
          borderRadius: BorderRadius.circular(8.0), // Rounded corners
          border: Border.all(
            color: borderColor,
            width: 1,
          ),
        ),
        child: Row(
          children: [
            Icon(
              Icons.chat_bubble_outline_rounded, // Icon similar to image (speech bubble)
              color: iconColor,
              size: 20,
            ),
            const SizedBox(width: 12), // Spacing between icon and text
            Expanded(
              child: Text(
                task.title, // Task title
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: theme.textTheme.bodyLarge?.copyWith( 
                  color: textColor,
                  fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
                ),
              ),
            ),
            if (selected)
              SizedBox(
                width: 36,
                height: 36,
                child: IconButton(
                  padding: EdgeInsets.zero,
                  splashRadius: 18, 
                  iconSize: 20,
                  icon: Icon(Icons.close, color: closeButtonColor), // closeButtonColor is now guaranteed to be initialized
                  onPressed: onDelete,
                ),
              ),
          ],
        ),
      ),
    );
  }
}