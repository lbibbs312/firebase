import 'dart:async';
// import 'dart:io'; // Keep for uploadArtifact if you plan to implement it, but it's not used in download
import 'dart:typed_data';
import 'package:auto_gpt_flutter_client/models/step_request_body.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';
import 'package:flutter/foundation.dart' show kIsWeb; // For platform check
// REMOVE: import 'dart:html' as html; // We will avoid direct html import here for now
import 'package:flutter/foundation.dart' show kDebugMode;
// Add for cross-platform URL launching
import 'package:url_launcher/url_launcher.dart';

// If you decide to save directly to device storage on mobile:
// import 'package:path_provider/path_provider.dart';
// import 'package:http/http.dart' as http; // Or package:dio
// import 'dart:io' as io; // For File operations, aliased to avoid conflict if any

class ChatService {
  final RestApiUtility api;

  ChatService(this.api);

  Stream<Map<String, dynamic>> streamStepExecution(
      String taskId, String? userInput) {
    String endpointPath = 'agent/tasks/$taskId/steps';
    if (userInput != null && userInput.isNotEmpty) {
      endpointPath += '?input=${Uri.encodeComponent(userInput)}';
    }
    final String fullSseUrl = '${api.agentBaseUrl}/$endpointPath';
    try {
      return api.streamEvents(fullSseUrl);
    } catch (e) {
      if (kDebugMode) {
        print("ChatService: Error initiating SSE stream for step execution: $e");
      }
      return Stream.error(e);
    }
  }

  Future<Map<String, dynamic>> executeStep(
      String taskId, StepRequestBody stepRequestBody) async {
    if (kDebugMode) {
      print(
          "ChatService WARNING: executeStep (single JSON response) called. This is likely deprecated if using SSE for chat.");
    }
    try {
      return await api.post(
          'agent/tasks/$taskId/steps', stepRequestBody.toJson());
    } catch (e) {
      rethrow;
    }
  }

  Future<Map<String, dynamic>> getStepDetails(
      String taskId, String stepId) async {
    try {
      return await api.get('agent/tasks/$taskId/steps/$stepId');
    } catch (e) {
      throw Exception('ChatService: Failed to get step details: $e');
    }
  }

  Future<Map<String, dynamic>> listTaskSteps(String taskId,
      {int currentPage = 1, int pageSize = 10}) async {
    try {
      return await api.get(
          'agent/tasks/$taskId/steps?current_page=$currentPage&page_size=$pageSize');
    } catch (e) {
      throw Exception('ChatService: Failed to list task steps: $e');
    }
  }

  Future<Map<String, dynamic>> uploadArtifact(
      String taskId, dynamic /* File or web File */ artifactFile, String uri) async {
    // Proper implementation needs to handle File for mobile and html.File for web.
    // This is a placeholder.
    if (kDebugMode) {
      print(
          "ChatService uploadArtifact: Not fully implemented for cross-platform or web file handling yet.");
    }
    return Future.value({'status': 'Upload not fully implemented yet'});
  }

  Future<void> downloadArtifact(String taskId, String artifactId) async {
    // Construct the full URL to the artifact download endpoint.
    // This assumes your RestApiUtility.baseUrl is the base of your API
    // and the path 'agent/tasks/$taskId/artifacts/$artifactId' directly serves the file
    // with appropriate Content-Disposition headers for download.
    final String downloadUrlString = "${api.agentBaseUrl}/agent/tasks/$taskId/artifacts/$artifactId";
    final Uri downloadUri = Uri.parse(downloadUrlString);

    try {
      if (kIsWeb) {
        // For web, launching the URL directly will trigger the browser's download mechanism
        // if the server sends the correct headers (e.g., Content-Disposition: attachment).
        // This avoids needing dart:html directly for the download trigger here.
        if (await canLaunchUrl(downloadUri)) {
          await launchUrl(downloadUri, webOnlyWindowName: '_blank');
        } else {
          throw Exception('Could not launch $downloadUri for web download.');
        }
      } else {
        // For mobile (Android/iOS)
        if (await canLaunchUrl(downloadUri)) {
          // This will open the URL in the default browser or prompt the user.
          // The browser will handle the actual download.
          // For more control (e.g., saving to a specific path, progress),
          // you'd use path_provider + http/dio to fetch bytes and save to a file.
          await launchUrl(downloadUri, mode: LaunchMode.externalApplication);
        } else {
          throw Exception('Could not launch $downloadUri for mobile download.');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('ChatService: An error occurred while downloading the artifact: $e');
      }
      // Re-throw or handle as appropriate for your UI
      throw Exception(
          'ChatService: An error occurred while downloading the artifact: $e');
    }
  }
}