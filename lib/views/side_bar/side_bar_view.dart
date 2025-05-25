// lib/views/side_bar/side_bar_view.dart
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:fluttertoast/fluttertoast.dart'; // For error feedback on URL launch
// Assuming AppColors is in this path, adjust if necessary
import 'package:auto_gpt_flutter_client/constants/app_colors.dart';

class SideBarView extends StatelessWidget {
  final ValueNotifier<String> selectedViewNotifier;
  final bool isCollapsed;
  final VoidCallback onToggleCollapse;
  final double expandedWidth;
  final double collapsedWidth;

  const SideBarView({
    super.key,
    required this.selectedViewNotifier,
    required this.isCollapsed,
    required this.onToggleCollapse,
    this.expandedWidth = 200.0,
    this.collapsedWidth = 60.0,
  });

  void _launchURL(String urlString) async {
    var url = Uri.parse(urlString);
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      Fluttertoast.showToast(msg: 'Could not launch $urlString');
    }
  }

  Widget _buildSidebarItem({
    required BuildContext context,
    required IconData icon,
    required String label,
    required String viewName,
    required String selectedView,
    required bool isBenchmarkRunning,
    required bool isCollapsed,
  }) {
    final bool isSelected = selectedView == viewName;
    final theme = Theme.of(context);
    // Use colorScheme for more semantic color choices
    final Color selectedColor = theme.colorScheme.primary;
    final Color unselectedColor = theme.colorScheme.onSurface; // General text/icon color on surface

    final iconColor = isSelected ? selectedColor : unselectedColor.withOpacity(0.7);
    final textColor = isSelected ? selectedColor : unselectedColor;

    return Material(
      color: Colors.transparent, // InkWell parent should be transparent
      child: InkWell(
        onTap: isBenchmarkRunning && viewName != 'SettingsView' // Allow settings even if benchmark running
            ? null
            : () => selectedViewNotifier.value = viewName,
        borderRadius: BorderRadius.circular(8), // Consistent border radius
        splashColor: selectedColor.withOpacity(0.12),
        highlightColor: selectedColor.withOpacity(0.08),
        child: Container(
          width: double.infinity,
          padding: EdgeInsets.symmetric(
            vertical: isCollapsed ? 12 : 10,
            horizontal: isCollapsed ? 0 : 16,
          ),
          margin: EdgeInsets.symmetric(horizontal: isCollapsed ? 6 : 8, vertical: 4), // Adjusted margins
          decoration: BoxDecoration(
            // Use a slightly more prominent background for selected item
            color: isSelected ? selectedColor.withOpacity(0.15) : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            mainAxisAlignment: isCollapsed ? MainAxisAlignment.center : MainAxisAlignment.start,
            children: [
              Icon(icon, color: iconColor, size: 22), // Slightly smaller icon
              if (!isCollapsed) ...[
                const SizedBox(width: 12), // Slightly smaller gap
                Expanded(
                  child: Text(
                    label,
                    style: TextStyle(
                      color: textColor,
                      fontWeight: isSelected ? FontWeight.bold : FontWeight.normal, // Bolder when selected
                      fontSize: 14,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ]
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildExternalLinkItem({
    required BuildContext context,
    required IconData? iconData,
    required String? imageAsset,
    required String tooltip,
    required String url,
    required bool isCollapsed,
  }) {
    final theme = Theme.of(context);
    final Color iconColor = theme.colorScheme.onSurface.withOpacity(0.65); // Consistent icon color
    final Color textColor = theme.colorScheme.onSurface.withOpacity(0.85); // Consistent text color

    Widget iconWidget;
    if (imageAsset != null) {
      // For image assets, consider providing a way to theme them if they are SVGs
      // or ensure they look good on dark backgrounds.
      iconWidget = Image.asset(imageAsset, width: isCollapsed ? 22 : 18, height: isCollapsed ? 22 : 18);
    } else {
      iconWidget = Icon(iconData ?? Icons.link, size: isCollapsed ? 22 : 20, color: iconColor);
    }

    return Tooltip(
      message: tooltip,
      preferBelow: true,
      textStyle: TextStyle(color: theme.colorScheme.onInverseSurface), // Text color for tooltip
      decoration: BoxDecoration( // Tooltip background
        color: theme.colorScheme.inverseSurface,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => _launchURL(url),
          borderRadius: BorderRadius.circular(isCollapsed ? 100 : 8),
          splashColor: theme.colorScheme.primary.withOpacity(0.12),
          highlightColor: theme.colorScheme.primary.withOpacity(0.08),
          child: Container(
            width: double.infinity,
            padding: EdgeInsets.symmetric(
              vertical: isCollapsed ? 12 : 10,
              horizontal: isCollapsed ? 0 : 16,
            ),
            margin: EdgeInsets.symmetric(horizontal: isCollapsed ? 6 : 8, vertical: 4),
            child: isCollapsed
                ? Center(child: iconWidget)
                : Row(
                    children: [
                      iconWidget,
                      const SizedBox(width: 12),
                      Expanded(child: Text(tooltip, style: TextStyle(fontSize: 13, color: textColor))),
                    ],
                  ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final taskQueueViewModel = Provider.of<TaskQueueViewModel>(context, listen: true);
    final settingsViewModel = Provider.of<SettingsViewModel>(context, listen: true);
    final theme = Theme.of(context);

    return Material(
      elevation: 1.0,
      // Use canvasColor from the theme for the sidebar background
      color: theme.canvasColor,
      child: ValueListenableBuilder(
          valueListenable: selectedViewNotifier,
          builder: (context, String selectedView, _) {
            return SizedBox(
              height: double.infinity,
              child: Column(
                children: [
                  Material(
                    color: Colors.transparent,
                    child: InkWell(
                      onTap: onToggleCollapse,
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(vertical: 16.0),
                        child: Icon(
                          isCollapsed ? Icons.arrow_forward_ios_rounded : Icons.arrow_back_ios_new_rounded,
                          size: 18,
                          // Use a color from the theme for the toggle icon
                          color: theme.colorScheme.onSurface.withOpacity(0.6),
                        ),
                      ),
                    ),
                  ),
                  // Use Divider with theme color
                  Divider(height: 1, thickness: 1, color: theme.dividerColor.withOpacity(0.5)),

                  _buildSidebarItem(
                    context: context,
                    icon: Icons.chat_bubble_outline_rounded,
                    label: "Chat / Tasks",
                    viewName: 'TaskView',
                    selectedView: selectedView,
                    isBenchmarkRunning: taskQueueViewModel.isBenchmarkRunning,
                    isCollapsed: isCollapsed,
                  ),
                  if (settingsViewModel.isDeveloperModeEnabled)
                    _buildSidebarItem(
                      context: context,
                      icon: Icons.emoji_events_outlined,
                      label: "Skill Tree",
                      viewName: 'SkillTreeView',
                      selectedView: selectedView,
                      isBenchmarkRunning: taskQueueViewModel.isBenchmarkRunning,
                      isCollapsed: isCollapsed,
                    ),
                  _buildSidebarItem(
                    context: context,
                    icon: Icons.settings_outlined,
                    label: "Settings",
                    viewName: 'SettingsView',
                    selectedView: selectedView,
                    // Settings can always be accessed
                    isBenchmarkRunning: false, // Ensure settings is always clickable from a benchmark perspective
                    isCollapsed: isCollapsed,
                  ),
                  const Spacer(),
                  Divider(height: 1, thickness: 1, color: theme.dividerColor.withOpacity(0.5)),
                   Padding(
                     padding: EdgeInsets.symmetric(vertical: isCollapsed ? 6.0 : 8.0), // Adjusted padding
                     child: Column(
                      children: [
                        _buildExternalLinkItem(
                          context: context,
                          iconData: Icons.book_outlined,
                          imageAsset: null,
                          tooltip: 'Documentation',
                          url: 'https://aiedge.medium.com/autogpt-forge-e3de53cc58ec',
                          isCollapsed: isCollapsed,
                        ),
                        _buildExternalLinkItem(
                          context: context,
                          iconData: null,
                          imageAsset: 'assets/images/discord_logo.png', // Ensure this asset looks good on dark bg
                          tooltip: 'Join Discord',
                          url: 'https://discord.gg/autogpt',
                          isCollapsed: isCollapsed,
                        ),
                        _buildExternalLinkItem(
                          context: context,
                          iconData: null,
                          imageAsset: 'assets/images/twitter_logo.png', // Ensure this asset looks good on dark bg
                          tooltip: 'Follow on X',
                          url: 'https://twitter.com/Auto_GPT',
                          isCollapsed: isCollapsed,
                        ),
                      ],
                     ),
                   ),
                ],
              ),
            );
          }),
    );
  }
}
