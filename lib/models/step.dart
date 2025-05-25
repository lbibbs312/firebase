// In lib/models/step.dart
// TODO: Refactor this to match which values are required and optional (as per your original TODO)
import 'package:auto_gpt_flutter_client/models/artifact.dart';

class Step {
  final String input; // Will be '' if not provided by backend
  final Map<String, dynamic> additionalInput;
  final String taskId;
  final String stepId;
  final String name; // Consider if this can be nullable or have a default
  final String status; // Consider if this can be nullable or have a default
  final String output; // Will be '' if not provided by backend
  final Map<String, dynamic> additionalOutput;
  final List<Artifact> artifacts; // Kept as non-nullable, will default to empty list
  final bool isLast;

  Step({
    required this.input,
    required this.additionalInput,
    required this.taskId,
    required this.stepId,
    required this.name,
    required this.status,
    required this.output,
    required this.additionalOutput,
    required this.artifacts, // artifacts is required in the constructor
    required this.isLast,
  });

  factory Step.fromMap(Map<String, dynamic>? map) {
    if (map == null) {
      // Or return a default Step / throw a more specific error
      // For now, to match existing behavior of throwing if map is null.
      throw ArgumentError('Null map provided to Step.fromMap');
    }
    
    // Safely parse artifacts
    List<Artifact> parsedArtifacts = [];
    if (map['artifacts'] != null && map['artifacts'] is List) {
      try {
        parsedArtifacts = (map['artifacts'] as List)
            .map((artifact) {
              if (artifact is Map<String, dynamic>) {
                return Artifact.fromJson(artifact);
              }
              // Handle cases where an item in the list isn't a map, or log/skip
              print("Step.fromMap: Warning - artifact in list is not a valid map: $artifact");
              return null; // Or some default/error Artifact placeholder
            })
            .whereType<Artifact>() // Filter out any nulls if an artifact was invalid
            .toList();
      } catch (e) {
        print("Step.fromMap: Error parsing artifacts: $e. Defaulting to empty list.");
        // parsedArtifacts will remain empty if an error occurs
      }
    } else if (map['artifacts'] != null) {
        print("Step.fromMap: Warning - 'artifacts' field is present but not a List. Defaulting to empty list. Value: ${map['artifacts']}");
    }


    return Step(
      input: map['input']?.toString() ?? '', // Ensure it's a string
      additionalInput: map['additional_input'] != null && map['additional_input'] is Map
          ? Map<String, dynamic>.from(map['additional_input'])
          : {},
      taskId: map['task_id']?.toString() ?? '',
      stepId: map['step_id']?.toString() ?? '',
      name: map['name']?.toString() ?? '',
      status: map['status']?.toString() ?? '',
      output: map['output']?.toString() ?? '',
      additionalOutput: map['additional_output'] != null && map['additional_output'] is Map
          ? Map<String, dynamic>.from(map['additional_output'])
          : {},
      artifacts: parsedArtifacts, // Use the safely parsed list
      isLast: map['is_last'] as bool? ?? false, // Ensure isLast is treated as bool
    );
  }
}