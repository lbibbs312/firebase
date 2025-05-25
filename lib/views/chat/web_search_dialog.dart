// File: auto_gpt_flutter_client/lib/views/chat/web_search_dialog.dart
// (Or auto_gpt_flutter_client/lib/widgets/dialogs/web_search_dialog.dart)

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:url_launcher/url_launcher.dart';

// Adjust the import path based on where you placed web_search_result.dart
import 'package:auto_gpt_flutter_client/viewmodels/web_search_result.dart';
// OR: import 'package:auto_gpt_flutter_client/models/web_search_result.dart';


void showWebSearchOverlay(BuildContext context, WebSearchResult result) {
  final theme = Theme.of(context); // Get theme for consistent styling

  if (result.error != null && result.error!.isNotEmpty) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(children: [
          Icon(Icons.error_outline, color: Colors.red.shade700),
          SizedBox(width: 8),
          Text("Web Search Error")
        ]),
        content: Text(result.error!),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(context).pop(), child: Text("OK"))
        ],
      ),
    );
    return;
  }

  showDialog(
    context: context,
    builder: (BuildContext context) {
      return AlertDialog(
        title: Row(
          children: [
            Icon(Icons.travel_explore_rounded,
                color: theme.colorScheme.primary),
            SizedBox(width: 8),
            Text("Web Search Results"),
          ],
        ),
        contentPadding: EdgeInsets.fromLTRB(20, 16, 20, 0), // Adjusted padding
        content: Container(
          width: MediaQuery.of(context).size.width * 0.9, // Max 90% of screen width
          constraints: BoxConstraints(maxWidth: 700, maxHeight: MediaQuery.of(context).size.height * 0.75), // Max width & height
          child: Column( // Main column for summary and sources
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Summary:",
                  style: theme.textTheme.titleMedium
                      ?.copyWith(fontWeight: FontWeight.bold)),
              SizedBox(height: 8),
              Flexible( // Allows summary to scroll if too long, within the column's constraints
                child: SingleChildScrollView(
                  child: MarkdownBody(
                    data: result.summary.isEmpty ? "No summary provided." : result.summary,
                    selectable: true,
                    onTapLink: (text, href, title) {
                      if (href != null) {
                        final uri = Uri.parse(href);
                        canLaunchUrl(uri).then((canLaunch) {
                          if (canLaunch) launchUrl(uri, webOnlyWindowName: '_blank');
                        });
                      }
                    },
                  ),
                ),
              ),
              if (result.sources.isNotEmpty) ...[
                SizedBox(height: 20),
                Text("Sources:",
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.bold)),
                SizedBox(height: 8),
                Expanded( // Allows sources list to scroll if it exceeds available space
                  child: ListView.separated(
                    shrinkWrap: true, // Important when nested in non-scrolling parent that has constraints
                    itemCount: result.sources.length,
                    separatorBuilder: (context, index) => Divider(height: 1, indent: 16, endIndent: 16),
                    itemBuilder: (context, index) {
  final source = result.sources[index];
  return ListTile(
    // ... (leading and title) ...
    title: Text(
      source.title ?? Uri.parse(source.url).host,
      maxLines: 2,
      overflow: TextOverflow.ellipsis,
      style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600), // Changed to titleSmall for slightly smaller title
    ),
    subtitle: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          source.url,
          style: theme.textTheme.bodySmall?.copyWith(color: Colors.blueGrey.shade700), // Kept as bodySmall
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        if (source.description != null &&
            source.description!.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 5.0),
            child: Text(
              source.description!,
              style: theme.textTheme.bodySmall?.copyWith( // Use bodySmall here as well
                color: theme.textTheme.bodySmall?.color?.withOpacity(0.85), // Get color from bodySmall and adjust opacity
                height: 1.3
              ),
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
            ),
          ),
      ],
    ),
                        contentPadding: EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
                        isThreeLine: source.description != null && source.description!.isNotEmpty, // Adjust based on content
                        onTap: () async {
                          if (source.url.isEmpty) return;
                          final uri = Uri.parse(source.url);
                          if (await canLaunchUrl(uri)) {
                            await launchUrl(uri, webOnlyWindowName: '_blank');
                          } else {
                             if (context.mounted) { // Check if context is still valid
                                ScaffoldMessenger.of(context).showSnackBar(
                                 SnackBar(
                                     content: Text('Could not launch ${source.url}')),
                               );
                             }
                          }
                        },
                      );
                    },
                  ),
                ),
              ],
            ],
          ),
        ),
        actionsAlignment: MainAxisAlignment.center,
        actionsPadding: EdgeInsets.only(bottom: 12, top: 8),
        actions: <Widget>[
          TextButton(
            child: Text("Close", style: TextStyle(fontSize: 16)),
            onPressed: () {
              Navigator.of(context).pop();
            },
          ),
        ],
      );
    },
  );
}