// lib/main_layout.dart
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/settings/settings_view.dart';
import 'package:auto_gpt_flutter_client/views/side_bar/side_bar_view.dart'; // Corrected path if needed
import 'package:auto_gpt_flutter_client/views/skill_tree/skill_tree_view.dart';
import 'package:auto_gpt_flutter_client/views/task/task_view.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_view.dart';
import 'package:auto_gpt_flutter_client/views/task_queue/task_queue_view.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  _MainLayoutState createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  final ValueNotifier<String> _selectedViewNotifier = ValueNotifier('TaskView');
  bool _isSidebarCollapsed = false;

  final double _expandedSidebarWidth = 200.0;
  final double _collapsedSidebarWidth = 60.0;

  void _toggleSidebarCollapse() {
    setState(() {
      _isSidebarCollapsed = !_isSidebarCollapsed;
    });
  }

  @override
  void dispose() {
    _selectedViewNotifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    // It's often better to get ViewModels only when needed, or use listen:false if not rebuilding for their changes.
    // For simplicity in this example, getting them here.
    final taskViewModel = Provider.of<TaskViewModel>(context, listen: false);
    final settingsViewModel = Provider.of<SettingsViewModel>(context, listen: false);
    // SkillTreeViewModel will be accessed within the ValueListenableBuilder where it's used.

    double currentSidebarWidth = _isSidebarCollapsed ? _collapsedSidebarWidth : _expandedSidebarWidth;

    // Widths for the content panels (TaskView, SettingsView)
    double taskViewPanelWidth = 280.0;
    double settingsViewPanelWidth = 280.0;
    
    if (screenWidth > 800) { // Desktop layout
      double contentAreaWidth = screenWidth - currentSidebarWidth;
      // skillTreeViewWidth, testQueueViewWidth will be calculated inside the builder

      return Scaffold(
        body: Row(
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              curve: Curves.easeInOutCubic, // A nice curve for animation
              width: currentSidebarWidth,
              child: SideBarView(
                selectedViewNotifier: _selectedViewNotifier,
                isCollapsed: _isSidebarCollapsed,
                onToggleCollapse: _toggleSidebarCollapse,
                expandedWidth: _expandedSidebarWidth,
                collapsedWidth: _collapsedSidebarWidth,
              ),
            ),
            Expanded(
              child: ValueListenableBuilder<String>(
                valueListenable: _selectedViewNotifier,
                builder: (context, String selectedViewName, _) {
                  // Access SkillTreeViewModel inside the builder where it's relevant
                  // Use listen: true if this part of the UI should rebuild when SkillTreeViewModel changes.
                  // If only its methods are called, listen: false might be okay.
                  final skillTreeViewModel = Provider.of<SkillTreeViewModel>(context); 
                  
                  if (selectedViewName == 'TaskView') {
                    // skillTreeViewModel.resetState(); // Call if needed when switching views
                    double currentTaskPanelWidth = (contentAreaWidth < taskViewPanelWidth) ? contentAreaWidth : taskViewPanelWidth;
                    double chatViewWidth = contentAreaWidth - currentTaskPanelWidth;
                    if (chatViewWidth < 0) chatViewWidth = 0;

                    return Row(
                      children: [
                        SizedBox(
                            width: currentTaskPanelWidth,
                            child: TaskView(viewModel: taskViewModel)),
                        if (chatViewWidth > 50) // Only show ChatView if there's reasonable space
                          Expanded(child: const ChatView())
                        else if (chatViewWidth > 0) // If very small, maybe show placeholder or nothing visible
                           SizedBox(width: chatViewWidth) // Or a Container with a color for debugging
                      ],
                    );
                  } else if (selectedViewName == 'SettingsView') {
                    // skillTreeViewModel.resetState();
                    double currentSettingsPanelWidth = (contentAreaWidth < settingsViewPanelWidth) ? contentAreaWidth : settingsViewPanelWidth;
                    double chatViewWidth = contentAreaWidth - currentSettingsPanelWidth;
                    if (chatViewWidth < 0) chatViewWidth = 0;

                    return Row(
                      children: [
                        SizedBox(
                            width: currentSettingsPanelWidth,
                            child: SettingsView(viewModel: settingsViewModel)),
                        if (chatViewWidth > 50)
                          Expanded(child: const ChatView())
                        else if (chatViewWidth > 0)
                           SizedBox(width: chatViewWidth)
                      ],
                    );
                  } else { // Default to SkillTree-related view (assuming viewName == 'SkillTreeView')
                    double skillTreeViewPanelWidth;
                    double testQueueViewPanelWidth;

                    if (skillTreeViewModel.selectedNode != null) {
                      skillTreeViewPanelWidth = contentAreaWidth * 0.25;
                      testQueueViewPanelWidth = contentAreaWidth * 0.25;
                    } else {
                      skillTreeViewPanelWidth = contentAreaWidth * 0.5;
                      testQueueViewPanelWidth = 0; // No node selected, no queue view
                    }
                    
                    // Ensure panels don't overflow
                    if (skillTreeViewPanelWidth > contentAreaWidth) skillTreeViewPanelWidth = contentAreaWidth;
                    double remainingForChatAndQueue = contentAreaWidth - skillTreeViewPanelWidth;
                    if (testQueueViewPanelWidth > remainingForChatAndQueue) testQueueViewPanelWidth = remainingForChatAndQueue;


                    return Row(
                      children: [
                        if (skillTreeViewPanelWidth > 0) // Only show if there's width
                          SizedBox(
                              width: skillTreeViewPanelWidth,
                              child: SkillTreeView(viewModel: skillTreeViewModel)),
                        if (skillTreeViewModel.selectedNode != null && testQueueViewPanelWidth > 0)
                          SizedBox(
                              width: testQueueViewPanelWidth,
                              child: TaskQueueView()), // Assuming TaskQueueView() doesn't need a const
                        Expanded( // ChatView takes all remaining space
                            child: const ChatView()),
                      ],
                    );
                  }
                },
              ),
            ),
          ],
        ),
      );
    } else { // Mobile/Tablet layout (CupertinoTabScaffold)
      return CupertinoTabScaffold(
        tabBar: CupertinoTabBar(
          items: const <BottomNavigationBarItem>[
            BottomNavigationBarItem(icon: Icon(CupertinoIcons.square_list), label: 'Tasks'),
            BottomNavigationBarItem(icon: Icon(CupertinoIcons.chat_bubble_2_fill), label: 'Chat'),
            BottomNavigationBarItem(icon: Icon(CupertinoIcons.settings_solid), label: 'Settings'),
          ],
          onTap: (index) {
            // Optionally update _selectedViewNotifier for consistency if desktop view is also visible,
            // or handle mobile navigation state separately.
            // Example:
            // if (index == 0) _selectedViewNotifier.value = 'TaskView';
            // else if (index == 1) { /* Decide what Chat tab means for selectedViewNotifier if anything */ }
            // else if (index == 2) _selectedViewNotifier.value = 'SettingsView';
          },
        ),
        tabBuilder: (BuildContext context, int index) {
          Widget viewToBuild;
          switch (index) {
            case 0: viewToBuild = TaskView(viewModel: taskViewModel); break;
            case 1: viewToBuild = const ChatView(); break; // ChatView is likely self-contained or gets VM via Provider higher up
            case 2: viewToBuild = SettingsView(viewModel: settingsViewModel); break;
            default: viewToBuild = const Center(child: Text("Unknown Tab"));
          }
          return CupertinoTabView(builder: (context) {
            return CupertinoPageScaffold(
              // navigationBar: CupertinoNavigationBar(middle: Text('Page Title')), // Optional
              child: SafeArea(child: viewToBuild),
            );
          });
        },
      );
    }
  }
}