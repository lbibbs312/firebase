// File: auto_gpt_flutter_client/lib/viewmodels/web_search_result.dart
// (Or auto_gpt_flutter_client/lib/models/web_search_result.dart)

class WebSearchResult {
  final String summary;
  final List<WebSource> sources;
  final String? error;

  WebSearchResult({required this.summary, required this.sources, this.error});

  factory WebSearchResult.fromJson(Map<String, dynamic> json) {
    if (json.containsKey('error') && json['error'] != null) {
      return WebSearchResult(
        summary: '',
        sources: [],
        error: json['error'] as String?,
      );
    }
    var sourcesList = json['sources'] as List? ?? [];
    List<WebSource> sources = sourcesList
        .map((i) => WebSource.fromJson(i as Map<String, dynamic>))
        .toList();
    return WebSearchResult(
      summary: json['summary'] as String? ?? 'No summary available.',
      sources: sources,
      error: null,
    );
  }
}

class WebSource {
  final String url;
  final String? title;
  final String? description; // Added field for meta description/snippet

  WebSource({
    required this.url,
    this.title,
    this.description, // Added to constructor
  });

  factory WebSource.fromJson(Map<String, dynamic> json) {
    return WebSource(
      url: json['url'] as String? ?? '',
      title: json['title'] as String?,
      description: json['description'] as String?, // Parse from JSON
    );
  }

  String get faviconUrl {
    if (url.isEmpty) {
      return '';
    }
    try {
      final uri = Uri.parse(url);
      if (uri.host.isEmpty) {
        return '';
      }
      return 'https://www.google.com/s2/favicons?sz=64&domain_url=${uri.host}';
    } catch (e) {
      return '';
    }
  }
}