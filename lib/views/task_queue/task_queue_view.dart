import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_status.dart';
import 'package:auto_gpt_flutter_client/models/test_option.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/task_queue/test_suite_button.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
// Assuming AppColors is in this path, adjust if necessary
import 'package:auto_gpt_flutter_client/constants/app_colors.dart';

class TaskQueueView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final viewModel = Provider.of<TaskQueueViewModel>(context);
    final nodeHierarchy = viewModel.selectedNodeHierarchy ?? [];
    final theme = Theme.of(context); // Get the current theme

    return Material(
      // Use scaffoldBackgroundColor from the theme for the main background
      color: theme.scaffoldBackgroundColor,
      child: Column(
        children: [
          // The list of tasks (tiles)
          Expanded(
            child: ListView.builder(
              itemCount: nodeHierarchy.length,
              itemBuilder: (context, index) {
                final node = nodeHierarchy[index];

                Widget leadingWidget;
                Color statusColor;
                Color innerCircleColor = AppColors.darkerGreyUI; // A contrasting inner color for dark theme

                switch (viewModel.benchmarkStatusMap[node]) {
                  case null:
                  case BenchmarkTaskStatus.notStarted:
                    // Using a more subtle color for not started status
                    statusColor = theme.colorScheme.onSurface.withOpacity(0.3);
                    leadingWidget = CircleAvatar(
                      radius: 12,
                      backgroundColor: statusColor,
                      child: CircleAvatar(
                        radius: 6,
                        // Use a color that contrasts with the outer circle and fits the theme
                        backgroundColor: theme.colorScheme.surface,
                      ),
                    );
                    break;
                  case BenchmarkTaskStatus.inProgress:
                    leadingWidget = SizedBox(
                      width: 24,
                      height: 24,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        // CircularProgressIndicator will use theme.colorScheme.primary by default
                        // valueColor: AlwaysStoppedAnimation<Color>(theme.colorScheme.primary), // Explicitly if needed
                      ),
                    );
                    break;
                  case BenchmarkTaskStatus.success:
                    statusColor = AppColors.accentAffirmativeDark; // From your AppColors
                    leadingWidget = CircleAvatar(
                      radius: 12,
                      backgroundColor: statusColor,
                      child: Icon(Icons.check, size: 14, color: innerCircleColor),
                    );
                    break;
                  case BenchmarkTaskStatus.failure:
                    statusColor = AppColors.accentDeniedDark; // From your AppColors
                    leadingWidget = CircleAvatar(
                      radius: 12,
                      backgroundColor: statusColor,
                      child: Icon(Icons.close, size: 14, color: innerCircleColor),
                    );
                    break;
                }

                return Container(
                  margin: EdgeInsets.fromLTRB(20, 8, 20, 8), // Increased vertical margin slightly
                  padding: EdgeInsets.symmetric(vertical: 8, horizontal: 2), // Added some internal padding
                  decoration: BoxDecoration(
                    // Use cardColor or surface color from the theme for list item background
                    color: theme.cardColor,
                    // Use a subtle border color from the theme
                    border: Border.all(color: theme.colorScheme.onSurface.withOpacity(0.15), width: 1),
                    borderRadius: BorderRadius.circular(8), // Consistent rounded corners
                  ),
                  child: ListTile(
                    leading: leadingWidget,
                    // Text widgets will inherit color from ListTileThemeData or TextTheme
                    title: Text(
                      node.label,
                      textAlign: TextAlign.center,
                      style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600), // Using titleMedium for consistency
                    ),
                    subtitle: Text(
                      node.data.info.description,
                      textAlign: TextAlign.center,
                      style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurface.withOpacity(0.7)), // Using bodySmall for less emphasis
                    ),
                    // Ensure onTap is handled if these list items are interactive
                    // onTap: () { /* Handle tap if necessary */ },
                  ),
                );
              },
            ),
          ),

          // Buttons at the bottom
          Padding(
            padding: EdgeInsets.all(20),
            child: Column(
              children: [
                // TestSuiteButton should ideally also pick up theme styles
                // If TestSuiteButton internally uses ElevatedButton, it will use elevatedButtonTheme
                TestSuiteButton(
                  isDisabled: viewModel.isBenchmarkRunning,
                  selectedOptionString: viewModel.selectedOption.description,
                  onOptionSelected: (selectedOption) {
                    print('Option Selected: $selectedOption');
                    final skillTreeViewModel =
                        Provider.of<SkillTreeViewModel>(context, listen: false);
                    viewModel.updateSelectedNodeHierarchyBasedOnOption(
                        TestOptionExtension.fromDescription(selectedOption)!,
                        skillTreeViewModel.selectedNode,
                        skillTreeViewModel.skillTreeNodes,
                        skillTreeViewModel.skillTreeEdges);
                  },
                  onPlayPressed: (selectedOption) {
                    print('Starting benchmark with option: $selectedOption');
                    final chatViewModel =
                        Provider.of<ChatViewModel>(context, listen: false);
                    final taskViewModel =
                        Provider.of<TaskViewModel>(context, listen: false);
                    chatViewModel.clearCurrentTaskAndChats();
                    viewModel.runBenchmark(chatViewModel, taskViewModel);
                  },
                ),
                SizedBox(height: 8), // Gap of 8 points between buttons
              ],
            ),
          ),
        ],
      ),
    );
  }
}
